package com.examscanner.scanner

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max

object PageDetector {
    fun detect(gray: Mat): QuadD? {
        val width = gray.cols()
        val height = gray.rows()
        if (width < 32 || height < 32) return null
        val imageArea = width.toDouble() * height.toDouble()
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        var best: QuadD? = null
        var bestArea = 0.0
        try {
            for (flag in intArrayOf(Imgproc.THRESH_BINARY, Imgproc.THRESH_BINARY_INV)) {
                val binary = Mat()
                Imgproc.threshold(blurred, binary, 0.0, 255.0, flag + Imgproc.THRESH_OTSU)
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
                    if (area < 0.08 * imageArea || area > 0.98 * imageArea) {
                        contour.release()
                        continue
                    }
                    val curve = MatOfPoint2f(*contour.toArray())
                    val approx = MatOfPoint2f()
                    Imgproc.approxPolyDP(
                        curve,
                        approx,
                        0.02 * Imgproc.arcLength(curve, true),
                        true,
                    )
                    if (approx.total() == 4L) {
                        val points = approx.toArray().toList()
                        val polygon = MatOfPoint(*approx.toArray())
                        val convex = points.size == 4 &&
                            distinct(points) &&
                            Imgproc.isContourConvex(polygon)
                        polygon.release()
                        if (convex) {
                            val ordered = order(points)
                            if (ordered != null && area > bestArea) {
                                bestArea = area
                                best = ordered
                            }
                        }
                    }
                    approx.release()
                    curve.release()
                    contour.release()
                }
                hierarchy.release()
                binary.release()
            }
        } finally {
            blurred.release()
        }
        if (best == null) return null
        val minimumSide = max(24.0, 0.08 * minOf(width, height))
        if (best.minSide() < minimumSide) return null
        return best
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

    private fun distinct(points: List<Point>): Boolean {
        for (first in points.indices) {
            for (second in first + 1 until points.size) {
                val a = points[first]
                val b = points[second]
                if (kotlin.math.hypot(a.x - b.x, a.y - b.y) < 1.0) return false
            }
        }
        return true
    }

    private fun toPointD(point: Point) = PointD(point.x, point.y)
}
