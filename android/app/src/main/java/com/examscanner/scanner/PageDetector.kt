package com.examscanner.scanner

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object PageDetector {
    private const val MIN_AREA_RATIO = 0.12
    private const val MAX_AREA_RATIO = 0.92
    private const val MIN_CORNER_DEG = 55.0
    private const val MAX_CORNER_DEG = 125.0

    fun detect(gray: Mat): QuadD? = detect(gray, 0)

    private fun detect(gray: Mat, depth: Int): QuadD? {
        // Inside a container the binaries carry thin bright bridges between the page and
        // its surroundings; an opening pass sized from the image cuts them while a
        // several-hundred-pixel page survives untouched.
        val openKernel = if (depth > 0) {
            val side = minOf(gray.cols(), gray.rows())
            ((side * 0.012).toInt() / 2 * 2 + 1).coerceIn(3, 31)
        } else {
            0
        }
        val direct = findBestQuad(gray, PAGE_CONFIG, openKernel)
        if (direct != null) {
            val minimumSide = max(24.0, 0.08 * minOf(gray.cols(), gray.rows()))
            if (direct.first.minSide() >= minimumSide) {
                return direct.first
            }
        }
        // Fallback for a page shown inside a bright container (a sheet image on a
        // monitor): flatten the container and look for the page inside it, where the
        // global threshold can finally separate paper from background.
        if (depth >= 1) return null
        val container = findBestQuad(gray, CONTAINER_CONFIG) ?: return fullFrameQuad(gray)
        val warped = flattenQuad(gray, container.first) ?: return fullFrameQuad(gray)
        val inner = try {
            detect(warped, depth + 1)
        } finally {
            warped.release()
        } ?: return fullFrameQuad(gray)
        return remapQuad(inner, container.first)
    }

    /**
     * Fallback for a sheet that fills the whole frame: its edges are off-screen, so no
     * page contour exists, but a bright band on all four borders proves paper reaches
     * every side and the full image is the page.
     */
    private fun fullFrameQuad(gray: Mat): QuadD? {
        val width = gray.cols()
        val height = gray.rows()
        if (width < 200 || height < 200) return null
        val band = max(4, (0.05 * minOf(width, height)).roundToInt())
        val top = gray.submat(0, band, 0, width)
        val bottom = gray.submat(height - band, height, 0, width)
        val left = gray.submat(0, height, 0, band)
        val right = gray.submat(0, height, width - band, width)
        val means = doubleArrayOf(
            Core.mean(top).`val`[0],
            Core.mean(bottom).`val`[0],
            Core.mean(left).`val`[0],
            Core.mean(right).`val`[0],
        )
        top.release()
        bottom.release()
        left.release()
        right.release()
        if (means.any { it < 150.0 }) return null
        val insetX = max(2.0, 0.01 * width)
        val insetY = max(2.0, 0.01 * height)
        return QuadD(
            PointD(insetX, insetY),
            PointD(width - 1 - insetX, insetY),
            PointD(width - 1 - insetX, height - 1 - insetY),
            PointD(insetX, height - 1 - insetY),
        )
    }

    private data class DetectConfig(
        val minAspect: Double,
        val maxAspect: Double,
        val minInteriorMean: Double,
        val minContrast: Double,
    )

    private val PAGE_CONFIG = DetectConfig(
        minAspect = 0.55,
        maxAspect = 0.88,
        minInteriorMean = 35.0,
        minContrast = 12.0,
    )

    /** Bright rectangles like a monitor screen; looser than a paper page. */
    private val CONTAINER_CONFIG = DetectConfig(
        minAspect = 0.35,
        maxAspect = 1.15,
        minInteriorMean = 90.0,
        minContrast = 20.0,
    )

    /** Higher is better, or null when no candidate passed the config's gates. */
    private fun findBestQuad(
        gray: Mat,
        config: DetectConfig,
        openKernel: Int = 0,
    ): Pair<QuadD, Double>? {
        val width = gray.cols()
        val height = gray.rows()
        if (width < 32 || height < 32) return null
        val imageArea = width.toDouble() * height.toDouble()
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        var best: QuadD? = null
        var bestScore = Double.NEGATIVE_INFINITY
        try {
            val binaries = buildBinaries(blurred)
            for (rawBinary in binaries) {
                var binary = rawBinary
                var owned: Mat? = null
                if (openKernel > 0) {
                    val kernel = Imgproc.getStructuringElement(
                        Imgproc.MORPH_RECT,
                        Size(openKernel.toDouble(), openKernel.toDouble()),
                    )
                    val opened = Mat()
                    Imgproc.morphologyEx(rawBinary, opened, Imgproc.MORPH_OPEN, kernel)
                    kernel.release()
                    owned = opened
                    binary = opened
                }
                val contours = ArrayList<MatOfPoint>()
                val hierarchy = Mat()
                Imgproc.findContours(
                    binary,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_SIMPLE,
                )
                for (contour in contours) {
                    val area = Imgproc.contourArea(contour)
                    if (area < MIN_AREA_RATIO * imageArea || area > MAX_AREA_RATIO * imageArea) {
                        contour.release()
                        continue
                    }
                    val curve = MatOfPoint2f(*contour.toArray())
                    val perimeter = Imgproc.arcLength(curve, true)
                    // Screen photos and slightly rounded paper corners defeat a single
                    // approximation tolerance, so several are tried per contour.
                    for (epsilonFactor in doubleArrayOf(0.015, 0.025, 0.04, 0.06)) {
                        val approx = MatOfPoint2f()
                        Imgproc.approxPolyDP(curve, approx, epsilonFactor * perimeter, true)
                        if (approx.total() == 4L) {
                            val points = approx.toArray().toList()
                            val polygon = MatOfPoint(*approx.toArray())
                            val convex = points.size == 4 &&
                                distinct(points) &&
                                Imgproc.isContourConvex(polygon)
                            polygon.release()
                            if (convex) {
                                val ordered = order(points)
                                if (ordered != null) {
                                    val score = scoreQuad(gray, ordered, area, imageArea, config)
                                    if (score > bestScore) {
                                        bestScore = score
                                        best = ordered
                                    }
                                }
                            }
                        }
                        approx.release()
                        if (bestScore > 0.0 && epsilonFactor >= 0.04) break
                    }
                    curve.release()
                    contour.release()
                }
                hierarchy.release()
                owned?.release()
                rawBinary.release()
            }
        } finally {
            blurred.release()
        }
        if (best == null || bestScore < 0.0) return null
        return best to bestScore
    }

    /** Maps a page quad found inside [container]'s rectangle back to original coords. */
    private fun remapQuad(inner: QuadD, container: QuadD): QuadD {
        val (width, height) = containerSize(container)
        val source = MatOfPoint2f(
            Point(container.tl.x, container.tl.y),
            Point(container.tr.x, container.tr.y),
            Point(container.br.x, container.br.y),
            Point(container.bl.x, container.bl.y),
        )
        val destination = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(width - 1.0, 0.0),
            Point(width - 1.0, height - 1.0),
            Point(0.0, height - 1.0),
        )
        val toRect = Imgproc.getPerspectiveTransform(source, destination)
        val toSource = Mat()
        Core.invert(toRect, toSource)
        val input = MatOfPoint2f(
            Point(inner.tl.x, inner.tl.y),
            Point(inner.tr.x, inner.tr.y),
            Point(inner.br.x, inner.br.y),
            Point(inner.bl.x, inner.bl.y),
        )
        val output = MatOfPoint2f()
        Core.perspectiveTransform(input, output, toSource)
        val mapped = output.toArray().toList()
        toRect.release()
        toSource.release()
        input.release()
        output.release()
        source.release()
        destination.release()
        return QuadD(
            PointD(mapped[0].x, mapped[0].y),
            PointD(mapped[1].x, mapped[1].y),
            PointD(mapped[2].x, mapped[2].y),
            PointD(mapped[3].x, mapped[3].y),
        )
    }

    private fun containerSize(quad: QuadD): Pair<Int, Int> {
        val top = quad.tl.distanceTo(quad.tr)
        val bottom = quad.bl.distanceTo(quad.br)
        val left = quad.tl.distanceTo(quad.bl)
        val right = quad.tr.distanceTo(quad.br)
        val width = ((top + bottom) / 2.0).roundToInt().coerceIn(120, 1200)
        val height = ((left + right) / 2.0).roundToInt().coerceIn(120, 1200)
        return width to height
    }

    private fun flattenQuad(gray: Mat, quad: QuadD): Mat? {
        val (width, height) = containerSize(quad)
        val source = MatOfPoint2f(
            Point(quad.tl.x, quad.tl.y),
            Point(quad.tr.x, quad.tr.y),
            Point(quad.br.x, quad.br.y),
            Point(quad.bl.x, quad.bl.y),
        )
        val destination = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(width - 1.0, 0.0),
            Point(width - 1.0, height - 1.0),
            Point(0.0, height - 1.0),
        )
        val transform = Imgproc.getPerspectiveTransform(source, destination)
        val warp = Mat()
        Imgproc.warpPerspective(
            gray,
            warp,
            transform,
            Size(width.toDouble(), height.toDouble()),
            Imgproc.INTER_LINEAR,
        )
        transform.release()
        source.release()
        destination.release()
        return warp
    }

    fun order(points: List<Point>): QuadD? {
        if (points.size != 4 || !distinct(points)) return null
        val sumMin = points.minByOrNull { it.x + it.y } ?: return null
        val sumMax = points.maxByOrNull { it.x + it.y } ?: return null
        val diffMin = points.minByOrNull { it.y - it.x } ?: return null
        val diffMax = points.maxByOrNull { it.y - it.x } ?: return null
        val tl = toPointD(sumMin)
        val br = toPointD(sumMax)
        val tr = toPointD(diffMin)
        val bl = toPointD(diffMax)
        val distinctCorners = listOf(tl, tr, br, bl).distinct()
        if (distinctCorners.size != 4) return null
        return QuadD(tl, tr, br, bl)
    }

    fun sharpness(gray: Mat): Double {
        if (gray.empty()) return 0.0
        val laplacian = Mat()
        return try {
            Imgproc.Laplacian(gray, laplacian, CvType.CV_64F)
            val mean = MatOfDouble()
            val std = MatOfDouble()
            Core.meanStdDev(laplacian, mean, std)
            val value = std.get(0, 0)[0]
            mean.release()
            std.release()
            value * value
        } finally {
            laplacian.release()
        }
    }

    fun brightness(gray: Mat): Double = Core.mean(gray).`val`[0]

    fun glareRatio(gray: Mat): Double {
        val bright = Mat()
        return try {
            Imgproc.threshold(gray, bright, 250.0, 255.0, Imgproc.THRESH_BINARY)
            val count = Core.countNonZero(bright).toDouble()
            count / (gray.cols().toDouble() * gray.rows().toDouble())
        } finally {
            bright.release()
        }
    }

    private fun buildBinaries(blurred: Mat): List<Mat> {
        val binaries = ArrayList<Mat>(4)
        for (flag in intArrayOf(Imgproc.THRESH_BINARY, Imgproc.THRESH_BINARY_INV)) {
            val otsu = Mat()
            Imgproc.threshold(blurred, otsu, 0.0, 255.0, flag + Imgproc.THRESH_OTSU)
            binaries.add(otsu)
        }
        val adaptive = Mat()
        Imgproc.adaptiveThreshold(
            blurred,
            adaptive,
            255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY,
            31,
            5.0,
        )
        binaries.add(adaptive)
        val adaptiveInv = Mat()
        Core.bitwise_not(adaptive, adaptiveInv)
        binaries.add(adaptiveInv)
        val edges = Mat()
        Imgproc.Canny(blurred, edges, 50.0, 150.0)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val closed = Mat()
        Imgproc.morphologyEx(edges, closed, Imgproc.MORPH_CLOSE, kernel)
        edges.release()
        kernel.release()
        binaries.add(closed)
        return binaries
    }


    /**
     * Higher is better. Negative means the candidate is rejected.
     * Favours paper-like aspect, near-right angles, bright interior and strong contrast.
     */
    private fun scoreQuad(
        gray: Mat,
        quad: QuadD,
        contourArea: Double,
        imageArea: Double,
        config: DetectConfig,
    ): Double {
        if (!anglesLookRectangular(quad)) return -1.0
        val aspect = pageAspect(quad) ?: return -1.0
        if (aspect < config.minAspect || aspect > config.maxAspect) return -1.0
        val areaRatio = contourArea / imageArea
        if (areaRatio < MIN_AREA_RATIO || areaRatio > MAX_AREA_RATIO) return -1.0
        val interior = interiorStats(gray, quad) ?: return -1.0
        if (interior.mean < config.minInteriorMean) return -1.0
        // A dark sheet on a bright desk reads as inside-darker; only absolute contrast
        // separates a real page from a uniform background.
        if (abs(interior.contrast) < config.minContrast) return -1.0
        val aspectMid = (config.minAspect + config.maxAspect) / 2.0
        val aspectSpread = (config.maxAspect - config.minAspect) / 2.0
        val aspectScore = 1.0 - abs(aspect - aspectMid) / (aspectSpread * 2.0)
        val areaScore = 1.0 - abs(areaRatio - 0.55) / 0.45
        val brightnessScore = ((interior.mean - config.minInteriorMean) / 80.0).coerceIn(0.0, 1.0)
        val contrastScore = (abs(interior.contrast) / 40.0).coerceIn(0.0, 1.0)
        return 2.5 * aspectScore + 1.5 * areaScore + 2.0 * brightnessScore + 1.5 * contrastScore
    }

    private fun pageAspect(quad: QuadD): Double? {
        val top = quad.tl.distanceTo(quad.tr)
        val bottom = quad.bl.distanceTo(quad.br)
        val left = quad.tl.distanceTo(quad.bl)
        val right = quad.tr.distanceTo(quad.br)
        val width = (top + bottom) / 2.0
        val height = (left + right) / 2.0
        if (width < 1.0 || height < 1.0) return null
        return min(width, height) / max(width, height)
    }

    private fun anglesLookRectangular(quad: QuadD): Boolean {
        val points = quad.points()
        for (index in points.indices) {
            val previous = points[(index + 3) % 4]
            val current = points[index]
            val next = points[(index + 1) % 4]
            val degrees = cornerDegrees(previous, current, next) ?: return false
            if (degrees < MIN_CORNER_DEG || degrees > MAX_CORNER_DEG) return false
        }
        return true
    }

    private fun cornerDegrees(previous: PointD, current: PointD, next: PointD): Double? {
        val ax = previous.x - current.x
        val ay = previous.y - current.y
        val bx = next.x - current.x
        val by = next.y - current.y
        val lengthA = hypot(ax, ay)
        val lengthB = hypot(bx, by)
        if (lengthA < 1.0 || lengthB < 1.0) return null
        val cos = ((ax * bx + ay * by) / (lengthA * lengthB)).coerceIn(-1.0, 1.0)
        return Math.toDegrees(acos(cos))
    }

    private data class InteriorStats(val mean: Double, val contrast: Double)

    private fun interiorStats(gray: Mat, quad: QuadD): InteriorStats? {
        val mask = Mat.zeros(gray.size(), CvType.CV_8UC1)
        return try {
            val polygon = MatOfPoint(
                Point(quad.tl.x, quad.tl.y),
                Point(quad.tr.x, quad.tr.y),
                Point(quad.br.x, quad.br.y),
                Point(quad.bl.x, quad.bl.y),
            )
            Imgproc.fillConvexPoly(mask, polygon, Scalar(255.0))
            polygon.release()
            val meanInside = Core.mean(gray, mask).`val`[0]
            Core.bitwise_not(mask, mask)
            val meanOutside = Core.mean(gray, mask).`val`[0]
            InteriorStats(meanInside, meanInside - meanOutside)
        } finally {
            mask.release()
        }
    }

    private fun distinct(points: List<Point>): Boolean {
        for (first in points.indices) {
            for (second in first + 1 until points.size) {
                val a = points[first]
                val b = points[second]
                if (hypot(a.x - b.x, a.y - b.y) < 1.0) return false
            }
        }
        return true
    }

    private fun toPointD(point: Point) = PointD(point.x, point.y)
}
