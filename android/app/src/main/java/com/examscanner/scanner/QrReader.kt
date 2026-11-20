package com.examscanner.scanner

import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.QRCodeDetector
import kotlin.math.roundToInt

/**
 * Reads the QR code printed in the sheet header.
 *
 * By the time this runs the page warp is upright, so the code always sits in the upper
 * left of the page. A cheap detect pass on that crop tells whether there is a code at
 * all; only then is the crop enlarged and decoded, because the code is a small part of
 * the sheet and the detector has far more trouble with it at page size. A binarized
 * retry covers washed-out prints.
 */
object QrReader {
    /** Share of the page that can hold the header code. */
    private const val SEARCH_WIDTH = 0.55
    private const val SEARCH_HEIGHT = 0.35
    /** Enlargement of the header crop while searching: the code is tiny at page size. */
    private const val ZOOM = 2.5

    /** Decoded text and, when the detector reports it, the four corners in page pixels. */
    data class Result(val text: String, val corners: List<Point>?)

    fun read(gray: Mat): Result? {
        if (gray.empty()) return null
        val width = gray.cols()
        val height = gray.rows()
        val detector = QRCodeDetector()
        val header = Mat(
            gray,
            Rect(
                0,
                0,
                (width * SEARCH_WIDTH).roundToInt().coerceIn(1, width),
                (height * SEARCH_HEIGHT).roundToInt().coerceIn(1, height),
            ),
        )
        val enlarged = Mat()
        val binary = Mat()
        try {
            // Detection is an order of magnitude cheaper than decoding, so a sheet with
            // no code in the header never pays for the enlarged attempts below.
            if (!detect(detector, header)) return null

            Imgproc.resize(header, enlarged, Size(), ZOOM, ZOOM, Imgproc.INTER_CUBIC)
            decode(detector, enlarged, ZOOM)?.let { return it }

            Imgproc.threshold(enlarged, binary, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
            decode(detector, binary, ZOOM)?.let { return it }

            // A code big enough to read at page size is rare but free to try.
            return decode(detector, header, 1.0)
        } catch (_: Throwable) {
            return null
        } finally {
            header.release()
            enlarged.release()
            binary.release()
        }
    }

    private fun detect(detector: QRCodeDetector, image: Mat): Boolean = try {
        val points = Mat()
        val found = detector.detect(image, points)
        points.release()
        found
    } catch (_: Throwable) {
        false
    }

    /**
     * Decodes one candidate image and maps the corners back to page pixels. The header
     * crop starts at the page origin, so [zoom] is all the mapping needed.
     */
    private fun decode(detector: QRCodeDetector, image: Mat, zoom: Double): Result? = try {
        val points = Mat()
        val text = detector.detectAndDecode(image, points)
        val corners = lookLikeQr(points)?.map { Point(it.x / zoom, it.y / zoom) }
        points.release()
        if (text.isNullOrBlank()) null else Result(text.trim(), corners)
    } catch (_: Throwable) {
        null
    }

    /** Four corner points of the code, in the coordinates of [points]; null if unreadable. */
    private fun lookLikeQr(points: Mat): List<Point>? {
        if (points.empty()) return null
        val count = points.rows() * points.cols()
        if (count < 4) return null
        val values = FloatArray(count * points.channels())
        points.get(0, 0, values)
        if (values.size < 8) return null
        return List(4) { Point(values[it * 2].toDouble(), values[it * 2 + 1].toDouble()) }
    }
}
