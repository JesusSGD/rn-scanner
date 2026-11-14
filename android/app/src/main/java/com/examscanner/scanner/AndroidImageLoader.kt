package com.examscanner.scanner

import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import kotlin.math.max

/**
 * Decodes a captured photo into a grayscale Mat for the scanner.
 *
 * The photo is never upscaled: anything above [MAX_SIDE] is reduced with area
 * interpolation, which keeps the scan fast without losing the bubble rings.
 */
object AndroidImageLoader {
    const val MAX_SIDE = 2200

    fun loadGray(path: String): Mat? {
        val image: Mat = Imgcodecs.imread(path, Imgcodecs.IMREAD_GRAYSCALE)
        if (image.empty()) {
            image.release()
            return null
        }
        val longSide = max(image.cols(), image.rows())
        if (longSide <= MAX_SIDE) return image
        val scale = MAX_SIDE.toDouble() / longSide
        val resized = Mat()
        try {
            Imgproc.resize(image, resized, Size(), scale, scale, Imgproc.INTER_AREA)
        } finally {
            image.release()
        }
        return resized
    }
}
