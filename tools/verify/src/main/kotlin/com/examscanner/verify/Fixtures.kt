package com.examscanner.verify

import com.examscanner.scanner.AnswerGrid
import com.examscanner.scanner.ExamTemplate
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.roundToInt

/**
 * Builds photo-like fixtures from the rectified sheet: the page is pasted on a dark desk,
 * tilted with a mild perspective and optionally rotated, blurred or darkened.
 */
object Fixtures {
    private val DESK = Scalar(88.0, 104.0, 122.0)

    data class Fixture(
        val name: String,
        val file: File,
        val expectOk: Boolean,
        val expectReason: String?,
        val expectBlocks: Int? = null,
        val expectAnswers: Int? = null,
    )

    fun build(sheetFile: File, template: ExamTemplate, outputDir: File): List<Fixture> {
        val sheet = Imgcodecs.imread(sheetFile.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)
        check(!sheet.empty()) { "Cannot read $sheetFile" }
        outputDir.mkdirs()

        val fixtures = ArrayList<Fixture>(7)
        val color = Mat()
        Imgproc.cvtColor(sheet, color, Imgproc.COLOR_GRAY2BGR)
        // The geometry of the rectified sheet is not exactly the template's (the page
        // contour lands a few pixels off), so the blanked regions are derived from the
        // lattice the scanner itself reads. A bright patch would shift Otsu's threshold,
        // hence the paper tone, sampled from blank paper right of the grid.
        val fit = AnswerGrid.fit(sheet, template)
        val paper = paperTone(sheet)

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
            fixtures.add(Fixture(name, file, true, null, expectBlocks = 3, expectAnswers = 33))
            composed.release()
            rotated.release()
        }

        // Sheets of the same family may print one to six A-E blocks, and only the last
        // block can be short. These fixtures blank the parts this photo did not print,
        // so the dynamic layout is exercised beyond the three-block template.
        if (fit != null) {
            fixtures.add(reduced(color, fit, paper, outputDir, "sheet_one_block", blocks = 1, lastRows = null))
            fixtures.add(reduced(color, fit, paper, outputDir, "sheet_two_blocks", blocks = 2, lastRows = null))
            fixtures.add(reduced(color, fit, paper, outputDir, "sheet_two_blocks_partial", blocks = 2, lastRows = 3))
            fixtures.add(
                reduced(
                    color,
                    fit,
                    paper,
                    outputDir,
                    "sheet_two_blocks_partial_flipped",
                    blocks = 2,
                    lastRows = 3,
                    flip = true,
                ),
            )
            fixtures.add(sixBlocks(color, fit, paper, outputDir))
        } else {
            println("fixtures skipped: the rectified sheet has no readable lattice")
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

    /**
     * A sheet with only the first [blocks] blocks printed and, when [lastRows] is set,
     * only that many top rows in the last block. Everything past the cut is blanked so
     * the detector has to recover the structure from the paper itself.
     */
    private fun reduced(
        sheet: Mat,
        fit: AnswerGrid.Fit,
        paper: Scalar,
        outputDir: File,
        name: String,
        blocks: Int,
        lastRows: Int?,
        flip: Boolean = false,
    ): Fixture {
        val blanked = sheet.clone()
        val radius = fit.radius
        val top = (fit.rows.first() - 2.2 * radius).toInt()
        val bottom = (fit.rows.last() + 2.2 * radius).toInt()
        val cut = (fit.columns[blocks * 5 - 1] + 1.5 * radius).toInt()
        fill(blanked, cut, top, blanked.cols(), bottom, paper)
        if (lastRows != null) {
            val below = ((fit.rows[lastRows - 1] + fit.rows[lastRows]) / 2.0).toInt()
            val left = (fit.columns[(blocks - 1) * 5] - 2.5 * radius).toInt()
            fill(blanked, left, below, cut, bottom, paper)
        }
        val mat = if (flip) {
            val rotated = Mat()
            Core.rotate(blanked, rotated, Core.ROTATE_180)
            blanked.release()
            rotated
        } else {
            blanked
        }
        val rows = (blocks - 1) * 15 + (lastRows ?: 15)
        return write(mat, outputDir, name, blocks, rows)
    }

    /**
     * A six-block sheet: the sheet interior is laid on a wider page, and the clean second
     * block is stamped at every later block position. This also tests the block origins
     * extrapolated past the template's three and the horizontal scale a wider page needs
     * when it is warped onto the template page.
     */
    private fun sixBlocks(
        sheet: Mat,
        fit: AnswerGrid.Fit,
        paper: Scalar,
        outputDir: File,
    ): Fixture {
        val radius = fit.radius
        val pitch = fit.columns[5] - fit.columns[0]
        val insetX = 6
        val insetY = 26
        val source = Mat(
            sheet,
            Rect(insetX, insetY, sheet.cols() - insetX - 14, sheet.rows() - insetY - 8),
        ).clone()
        val width = (fit.columns[9] + 4 * pitch + 3 * radius + 120 - insetX).toInt()
        val mat = Mat(source.rows(), width, sheet.type(), paper)
        val placed = Mat(mat, Rect(0, 0, source.cols(), source.rows()))
        source.copyTo(placed)
        placed.release()

        val left = (fit.columns[5] - 2.5 * radius - insetX).toInt()
        val right = (fit.columns[9] + 1.4 * radius - insetX).toInt()
        val top = (fit.rows.first() - 2.2 * radius - insetY).toInt()
        val bottom = (fit.rows.last() + 2.2 * radius - insetY).toInt()
        val block = Mat(source, Rect(left, top, right - left, bottom - top)).clone()
        fill(mat, (fit.columns[9] + 1.5 * radius - insetX).toInt(), top, mat.cols(), bottom, paper)
        for (k in 1..4) {
            val target = Mat(mat, Rect((left + k * pitch).roundToInt(), top, right - left, bottom - top))
            block.copyTo(target)
            target.release()
        }
        block.release()
        source.release()
        return write(mat, outputDir, "sheet_six_blocks", 6, 90)
    }

    private fun write(mat: Mat, outputDir: File, name: String, blocks: Int, rows: Int): Fixture {
        val file = File(outputDir, "$name.jpg")
        val composed = compose(mat)
        Imgcodecs.imwrite(file.absolutePath, composed, org.opencv.core.MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 92))
        composed.release()
        mat.release()
        return Fixture(name, file, true, null, expectBlocks = blocks, expectAnswers = rows)
    }

    private fun fill(mat: Mat, x0: Int, y0: Int, x1: Int, y1: Int, tone: Scalar) {
        val left = x0.coerceIn(0, mat.cols())
        val right = x1.coerceIn(left, mat.cols())
        val top = y0.coerceIn(0, mat.rows())
        val bottom = y1.coerceIn(top, mat.rows())
        if (right <= left || bottom <= top) return
        Imgproc.rectangle(
            mat,
            Point(left.toDouble(), top.toDouble()),
            Point(right - 1.0, bottom - 1.0),
            tone,
            Imgproc.FILLED,
        )
    }

    /** Mean tone of blank paper right of the answer grid, used for the blanked areas. */
    private fun paperTone(sheet: Mat): Scalar {
        val sample = Mat(sheet, Rect(950, 1100, 200, 250))
        val mean = Core.mean(sample).`val`[0]
        sample.release()
        return Scalar(mean, mean, mean)
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
