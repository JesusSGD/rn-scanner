package com.examscanner.scanner

import androidx.camera.core.ImageProxy
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.VisionCameraProxy
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Per-frame live detector used by the camera preview.
 *
 * It answers one question cheaply: "is there a well-lit, sharp, fully visible sheet
 * right now, and has it been still for a few frames?". The expensive reading only
 * runs once, on the captured photo.
 *
 * The returned array is intentionally made of primitives only:
 * `[status, ready, stableFrames, areaRatio, brightness, glare, sharpness, tlX, tlY,
 *   trX, trY, brX, brY, blX, blY]`.
 *
 * Quad corners are normalized 0..1 in **display** orientation (after applying
 * ImageProxy rotation), so JS can map them with `x * screenW`, `y * screenH`.
 */
class ScanPagePlugin(
    @Suppress("UNUSED_PARAMETER") proxy: VisionCameraProxy,
    @Suppress("UNUSED_PARAMETER") options: Map<String, Any>?,
) : FrameProcessorPlugin() {

    companion object {
        const val NAME = "scanPage"

        const val STATUS_SEARCHING = 0.0
        const val STATUS_TOO_DARK = 1.0
        const val STATUS_TOO_BRIGHT = 2.0
        const val STATUS_GLARE = 3.0
        const val STATUS_BLURRY = 4.0
        const val STATUS_TOO_SMALL = 5.0
        const val STATUS_OFF_FRAME = 6.0
        const val STATUS_READY = 7.0

        private const val TARGET_SIDE = 620
        /** Match a guide that fills most of the preview (~90% width). */
        private const val MIN_AREA = 0.28
        private const val MIN_BRIGHTNESS = 65.0
        private const val MAX_BRIGHTNESS = 235.0
        private const val MAX_GLARE = 0.10
        private const val MIN_SHARPNESS = 20.0
        private const val REQUIRED_STABLE_FRAMES = 4
        private const val STABILITY_RATIO = 0.015
    }

    private var rowBuffer = ByteArray(0)
    private var pixelBuffer = ByteArray(0)
    private var previousQuad: DoubleArray? = null
    private var stableFrames = 0

    override fun callback(frame: Frame, params: Map<String, Any>?): Any {
        val image = frame.imageProxy
        val gray = toGray(image) ?: return result(STATUS_SEARCHING, 0, 0.0, 0.0, 0.0, 0.0, 0.0, doubleArrayOf())
        return try {
            val width = gray.cols()
            val height = gray.rows()
            val brightness = PageDetector.brightness(gray)
            val glare = PageDetector.glareRatio(gray)
            val page = PageDetector.detect(gray)
            val quad = if (page != null) {
                normalizeQuad(page, width, height)
            } else {
                doubleArrayOf()
            }
            if (page == null) {
                resetStability()
                return result(STATUS_SEARCHING, 0, 0.0, 0.0, brightness, glare, 0.0, quad)
            }
            val areaRatio = page.area() / (width.toDouble() * height.toDouble())
            val sharpness = PageDetector.sharpness(gray)

            val status = when {
                brightness < MIN_BRIGHTNESS -> STATUS_TOO_DARK
                brightness > MAX_BRIGHTNESS -> STATUS_TOO_BRIGHT
                glare > MAX_GLARE -> STATUS_GLARE
                sharpness < MIN_SHARPNESS -> STATUS_BLURRY
                areaRatio < MIN_AREA -> STATUS_TOO_SMALL
                nearBorder(page, width, height) -> STATUS_OFF_FRAME
                else -> STATUS_READY
            }
            if (status != STATUS_READY) {
                resetStability()
                return result(status, 0, 0.0, areaRatio, brightness, glare, sharpness, quad)
            }

            val threshold = STABILITY_RATIO
            val previous = previousQuad
            val still = previous != null && maxCornerShift(previous, quad) <= threshold
            stableFrames = if (still) stableFrames + 1 else 1
            previousQuad = quad
            val ready = if (stableFrames >= REQUIRED_STABLE_FRAMES) 1 else 0
            return result(status, ready, stableFrames.toDouble(), areaRatio, brightness, glare, sharpness, quad)
        } finally {
            gray.release()
        }
    }

    private fun resetStability() {
        previousQuad = null
        stableFrames = 0
    }

    /** Corners must be a little away from the frame edge so the whole sheet is visible. */
    private fun nearBorder(page: QuadD, width: Int, height: Int): Boolean {
        val marginX = max(2.0, 0.015 * width)
        val marginY = max(2.0, 0.015 * height)
        return page.points().any {
            it.x <= marginX || it.y <= marginY || it.x >= width - marginX || it.y >= height - marginY
        }
    }

    private fun normalizeQuad(page: QuadD, width: Int, height: Int): DoubleArray {
        val w = width.toDouble().coerceAtLeast(1.0)
        val h = height.toDouble().coerceAtLeast(1.0)
        return doubleArrayOf(
            page.tl.x / w, page.tl.y / h,
            page.tr.x / w, page.tr.y / h,
            page.br.x / w, page.br.y / h,
            page.bl.x / w, page.bl.y / h,
        )
    }

    private fun maxCornerShift(previous: DoubleArray, current: DoubleArray): Double {
        if (previous.size != current.size) return Double.MAX_VALUE
        var shift = 0.0
        var index = 0
        while (index + 1 < current.size) {
            shift = max(shift, hypot(current[index] - previous[index], current[index + 1] - previous[index + 1]))
            index += 2
        }
        return shift
    }

    private fun result(
        status: Double,
        ready: Int,
        stable: Double,
        areaRatio: Double,
        brightness: Double,
        glare: Double,
        sharpness: Double,
        quad: DoubleArray,
    ): ArrayList<Any> {
        val values = ArrayList<Any>(16)
        values.add(status)
        values.add(ready.toDouble())
        values.add(stable)
        values.add(areaRatio)
        values.add(brightness)
        values.add(glare)
        values.add(sharpness)
        for (value in quad) values.add(value)
        return values
    }

    /**
     * Samples the luma plane into a small grayscale Mat in **display** orientation.
     * Sampling instead of resizing avoids copying the full-resolution frame on the hot path.
     */
    private fun toGray(image: ImageProxy): Mat? {
        val planes = image.planes
        if (planes.isEmpty()) return null
        val plane = planes[0]
        val width = image.width
        val height = image.height
        if (width <= 0 || height <= 0) return null

        val step = max(1, (max(width, height).toDouble() / TARGET_SIDE).roundToInt())
        val outWidth = (width + step - 1) / step
        val outHeight = (height + step - 1) / step
        val rowStride = plane.rowStride
        val pixelStride = max(1, plane.pixelStride)
        if (rowBuffer.size < rowStride) rowBuffer = ByteArray(rowStride)
        val size = outWidth * outHeight
        if (pixelBuffer.size < size) pixelBuffer = ByteArray(size)

        val buffer = plane.buffer
        var out = 0
        var y = 0
        while (y < height && out < size) {
            buffer.position(y * rowStride)
            val read = min(rowStride, buffer.remaining())
            buffer.get(rowBuffer, 0, read)
            var x = 0
            while (x < width && out < size) {
                pixelBuffer[out++] = rowBuffer[x * pixelStride]
                x += step
            }
            y += step
        }
        while (out < size) pixelBuffer[out++] = 0

        val sensorGray = Mat(outHeight, outWidth, CvType.CV_8UC1)
        sensorGray.put(0, 0, pixelBuffer)
        return rotateToDisplay(sensorGray, image.imageInfo.rotationDegrees)
    }

    /** Rotates sensor-buffer gray into the same orientation the preview shows. */
    private fun rotateToDisplay(sensorGray: Mat, rotationDegrees: Int): Mat {
        val flag = when (((rotationDegrees % 360) + 360) % 360) {
            90 -> Core.ROTATE_90_CLOCKWISE
            180 -> Core.ROTATE_180
            270 -> Core.ROTATE_90_COUNTERCLOCKWISE
            else -> {
                return sensorGray
            }
        }
        val rotated = Mat()
        Core.rotate(sensorGray, rotated, flag)
        sensorGray.release()
        return rotated
    }
}
