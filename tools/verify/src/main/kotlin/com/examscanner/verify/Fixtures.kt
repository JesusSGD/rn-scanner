package com.examscanner.verify

import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * Builds photo-like fixtures from the rectified sheet: the page is pasted on a dark desk,
 * tilted with a mild perspective and optionally rotated, blurred or darkened.
 */
object Fixtures {
    private val DESK = Scalar(88.0, 104.0, 122.0)

    data class Fixture(val name: String, val file: File, val expectOk: Boolean, val expectReason: String?)

    fun build(sheetFile: File, outputDir: File): List<Fixture> {
        val sheet = Imgcodecs.imread(sheetFile.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)
        check(!sheet.empty()) { "Cannot read $sheetFile" }
        outputDir.mkdirs()

        val fixtures = ArrayList<Fixture>(7)
        val color = Mat()
        Imgproc.cvtColor(sheet, color, Imgproc.COLOR_GRAY2BGR)

        val variants = listOf(
            "sheet_upright" to 0,
            "sheet_upside_down" to 180,
            "sheet_sideways" to 90,
            "sheet_sideways_other" to 270,
        )
        for ((name, angle) in variants) {
            val rotated = Mat()
            when (angle) {
                90 -> Core.rotate(color, rotated, Core.ROTATE_90_CLOCKWISE)
                180 -> Core.rotate(color, rotated, Core.ROTATE_180)
                270 -> Core.rotate(color, rotated, Core.ROTATE_90_COUNTERCLOCKWISE)
                else -> color.copyTo(rotated)
            }
            val composed = compose(rotated)
            val file = File(outputDir, "$name.jpg")
            Imgcodecs.imwrite(file.absolutePath, composed, org.opencv.core.MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 92))
            fixtures.add(Fixture(name, file, true, null))
            composed.release()
            rotated.release()
        }

        val blurred = Mat()
        Imgproc.GaussianBlur(color, blurred, Size(0.0, 0.0), 2.2)
        val blurFile = File(outputDir, "sheet_blurred.jpg")
        Imgcodecs.imwrite(blurFile.absolutePath, compose(blurred), org.opencv.core.MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 92))
        fixtures.add(Fixture("sheet_blurred", blurFile, false, "blurry"))
        blurred.release()

        val dark = Mat()
        color.convertTo(dark, org.opencv.core.CvType.CV_8UC3, 0.30, 0.0)
        val darkFile = File(outputDir, "sheet_dark.jpg")
        Imgcodecs.imwrite(darkFile.absolutePath, compose(dark), org.opencv.core.MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 92))
        fixtures.add(Fixture("sheet_dark", darkFile, false, "too_dark"))
        dark.release()

        val empty = Mat(700, 520, org.opencv.core.CvType.CV_8UC3, DESK)
        val emptyFile = File(outputDir, "no_sheet.jpg")
        Imgcodecs.imwrite(emptyFile.absolutePath, empty)
        fixtures.add(Fixture("no_sheet", emptyFile, false, "no_page"))
        empty.release()

        color.release()
        sheet.release()
        return fixtures
    }

    /** Pastes the sheet on a desk background with a mild handheld tilt. */
    private fun compose(sheet: Mat): Mat {
        val margin = 70
        val canvasHeight = sheet.rows() + margin * 2
        val canvasWidth = sheet.cols() + margin * 2

        val tilt = 16.0
        val source = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(sheet.cols() - 1.0, 0.0),
            Point(sheet.cols() - 1.0, sheet.rows() - 1.0),
            Point(0.0, sheet.rows() - 1.0),
        )
        val target = MatOfPoint2f(
            Point(margin + tilt, margin + 4.0),
            Point(margin + sheet.cols() - 1.0, margin + tilt * 0.5),
            Point(margin + sheet.cols() - 1.0 - tilt * 0.35, margin + sheet.rows() - 1.0),
            Point(margin.toDouble(), margin + sheet.rows() - 1.0 - tilt * 0.45),
        )
        val matrix = Imgproc.getPerspectiveTransform(source, target)
        val result = Mat(canvasHeight, canvasWidth, org.opencv.core.CvType.CV_8UC3)
        Imgproc.warpPerspective(
            sheet,
            result,
            matrix,
            Size(canvasWidth.toDouble(), canvasHeight.toDouble()),
            Imgproc.INTER_LINEAR,
            Core.BORDER_CONSTANT,
            DESK,
        )
        matrix.release()
        source.release()
        target.release()
        return result
    }
}
