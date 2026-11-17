package com.examscanner.scanner

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.Locale
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Locates the printed bubble lattice and reads the marked option of every question.
 *
 * The page warp only has to make the sheet roughly upright; the absolute position and
 * scale of the form inside the warp are recovered from the printed bubble rings, so the
 * reading does not depend on how exactly the page contour was found.
 */
object AnswerGrid {
    private const val BLOCKS = 3
    private const val PER_BLOCK = 5
    private const val COLUMNS = BLOCKS * PER_BLOCK
    private const val ROWS = 15
    private const val LETTERS = "ABCDE"

    data class Fit(
        val columns: DoubleArray,
        val rows: DoubleArray,
        val radius: Double,
        val matchedCells: Int,
        val totalCells: Int,
        val extraCells: Int,
        val matchedColumns: Int,
        val matchedRows: Int,
    ) {
        /** Fraction of template bubbles that sit on a detected printed ring. */
        val coverage: Double get() = if (totalCells == 0) 0.0 else matchedCells.toDouble() / totalCells

        /** Score used to pick the page orientation: real cells count, foreign rings do not. */
        val latticeScore: Double
            get() = matchedCells.toDouble() - 0.5 * extraCells +
                0.5 * (matchedColumns + matchedRows)

        /** Maps a fit computed on a downscaled page warp onto the full-scale warp. */
        fun scaled(factor: Double): Fit = copy(
            columns = DoubleArray(columns.size) { columns[it] * factor },
            rows = DoubleArray(rows.size) { rows[it] * factor },
            radius = radius * factor,
        )
    }

    private data class Circle(val x: Double, val y: Double, val r: Double)

    fun fit(gray: Mat, template: ExamTemplate, trace: ((String) -> Unit)? = null): Fit? {
        // Primary: recover the lattice from the printed rings. Fallback: lay the
        // template directly on the page warp and search a small global alignment —
        // the page warp itself is usually accurate enough for the overlay to land
        // on target even when ring detection fails.
        val circlesFit = fitFromCircles(gray, template, trace, useSoftMask = false)
        if (circlesFit != null && circlesFit.coverage >= 0.85) {
            return circlesFit
        }
        val directFit = fitFromTemplate(gray, template, trace)
        var best = when {
            circlesFit == null -> directFit
            directFit == null -> circlesFit
            directFit.latticeScore > circlesFit.latticeScore -> directFit
            else -> circlesFit
        }
        // Rescue path: binarized scans and very thin prints defeat both primaries,
        // while a strong blur turns their thin rings into soft disks that a third
        // mask recovers. It is gated behind a weak result because the extra mask
        // also admits noise circles that corrupt an otherwise good lattice.
        if (best == null || best.coverage < 0.75) {
            val softFit = fitFromCircles(gray, template, trace, useSoftMask = true)
            if (softFit != null && (best == null || softFit.latticeScore > best.latticeScore)) {
                best = softFit
            }
        }
        return best
    }

    /**
     * Lays the template lattice on the page warp and scores candidate global
     * alignments (scale + offset) by the ring darkness at the template positions.
     * Each template circle then sits at its own position without detecting rings.
     */
    private fun fitFromTemplate(gray: Mat, template: ExamTemplate, trace: ((String) -> Unit)?): Fit? {
        val width = gray.cols()
        val height = gray.rows()
        if (width < 200 || height < 200) return null
        val unit = width / 1241.0
        val templateColumns = DoubleArray(COLUMNS) { index ->
            val block = index / 5
            val letter = index % 5
            template.questions.first { it.number == block * 15 + 1 }.options[letter].x
        }
        val templateRows = DoubleArray(ROWS) { row ->
            template.questions.first { it.number == row + 1 }.options[0].y
        }

        val integral = Mat()
        Imgproc.integral(gray, integral, CvType.CV_32S)
        val stride = integral.cols()
        val data = IntArray(integral.rows() * stride)
        integral.get(0, 0, data)
        integral.release()
        val ringRadius = max(5, ((template.bubbleRadius + 1.0) * unit).roundToInt())
        val ring = ringKernel(ringRadius)
        val center = discKernel(3)

        // Which lattice cells actually exist: block 3 only has three rows, so the
        // lower rows of its five columns are blank paper on the sheet.
        val columnRowCount = IntArray(COLUMNS) { column -> if (column >= 10) 3 else 15 }
        val rowColumnCount = IntArray(ROWS) { row -> if (row < 3) 15 else 10 }

        fun evaluate(scaleX: Double, scaleY: Double, dx: Double, dy: Double): Triple<Int, DoubleArray, DoubleArray> {
            val columns = DoubleArray(COLUMNS) { scaleX * templateColumns[it] + dx }
            val rows = DoubleArray(ROWS) { scaleY * templateRows[it] + dy }
            var rings = 0
            for (column in 0 until COLUMNS) {
                for (row in 0 until columnRowCount[column]) {
                    val cx = columns[column].roundToInt()
                    val cy = rows[row].roundToInt()
                    val ringMean = boxMean(data, stride, width, height, cx, cy, ring)
                    val centerMean = boxMean(data, stride, width, height, cx, cy, center)
                    // A printed bubble reads as paper inside a dark ring; text or stray
                    // marks are dark through and through, so this contrast only holds
                    // for real rings.
                    if (centerMean > ringMean + 4.0) rings++
                }
            }
            return Triple(rings, columns, rows)
        }

        // Round 1: unit scale, offset search at a coarse step. The nested page quad can
        // land several pixels off, so the reach must exceed the local fine steps.
        var best = evaluate(1.0, 1.0, 0.0, 0.0)
        var bestOffsetX = 0.0
        var bestOffsetY = 0.0
        val coarseStep = max(1.0, (2.0 * unit).roundToInt().toDouble())
        val coarseReach = max(8.0 * unit, 0.02 * width)
        var offsetDy = -coarseReach
        while (offsetDy <= coarseReach) {
            var offsetDx = -coarseReach
            while (offsetDx <= coarseReach) {
                val candidate = evaluate(1.0, 1.0, offsetDx, offsetDy)
                if (candidate.first > best.first) {
                    best = candidate
                    bestOffsetX = offsetDx
                    bestOffsetY = offsetDy
                }
                offsetDx += coarseStep
            }
            offsetDy += coarseStep
        }

        // Round 2: fine scale and offset around the coarse winner.
        var bestScaleX = 1.0
        var bestScaleY = 1.0
        for (scaleX in doubleArrayOf(0.95, 0.975, 1.0, 1.025, 1.05)) {
            for (scaleY in doubleArrayOf(0.95, 0.975, 1.0, 1.025, 1.05)) {
                for (dy in doubleArrayOf(-2.0, -1.0, 0.0, 1.0, 2.0)) {
                    for (dx in doubleArrayOf(-2.0, -1.0, 0.0, 1.0, 2.0)) {
                        val candidate = evaluate(scaleX, scaleY, bestOffsetX + dx, bestOffsetY + dy)
                        if (candidate.first > best.first) {
                            best = candidate
                            bestScaleX = scaleX
                            bestScaleY = scaleY
                            bestOffsetX += dx
                            bestOffsetY += dy
                        }
                    }
                }
            }
        }

        val total = template.questions.sumOf { it.options.size }
        if (best.first < total * 0.5) {
            trace?.invoke("template fit rings=${best.first}")
            return null
        }
        val columns = best.second
        val rows = best.third
        val columnRings = IntArray(COLUMNS)
        val rowRings = IntArray(ROWS)
        var rings = 0
        for (column in 0 until COLUMNS) {
            for (row in 0 until columnRowCount[column]) {
                val cx = columns[column].roundToInt()
                val cy = rows[row].roundToInt()
                val ringMean = boxMean(data, stride, width, height, cx, cy, ring)
                val centerMean = boxMean(data, stride, width, height, cx, cy, center)
                if (centerMean > ringMean + 4.0) {
                    rings++
                    columnRings[column]++
                    rowRings[row]++
                }
            }
        }
        // The blank tail of block 3 (its columns below the third row) must be paper;
        // ring-like contrast there means the template is upside down or shifted, and
        // every such hit is evidence against this alignment.
        var blankHits = 0
        for (column in 10 until COLUMNS) {
            for (row in 3 until ROWS) {
                val cx = columns[column].roundToInt()
                val cy = rows[row].roundToInt()
                val ringMean = boxMean(data, stride, width, height, cx, cy, ring)
                val centerMean = boxMean(data, stride, width, height, cx, cy, center)
                if (centerMean > ringMean + 4.0) blankHits++
            }
        }
        trace?.invoke(
            "template fit rings=$rings blank=$blankHits cols=${columnRings.count { it > 0 }} scale=$bestScaleX/$bestScaleY offset=${bestOffsetX},${bestOffsetY}",
        )
        return Fit(
            columns = columns,
            rows = rows,
            radius = ringRadius.toDouble(),
            matchedCells = rings,
            totalCells = total,
            extraCells = 2 * blankHits,
            // Columns of block 3 only hold three cells, rows below the third hold ten.
            matchedColumns = columnRings.indices.count { columnRings[it] >= (columnRowCount[it] * 3) / 4 },
            matchedRows = rowRings.indices.count { rowRings[it] >= (rowColumnCount[it] * 3) / 4 },
        )
    }

    private fun fitFromCircles(
        gray: Mat,
        template: ExamTemplate,
        trace: ((String) -> Unit)?,
        useSoftMask: Boolean,
    ): Fit? {        val width = gray.cols()
        val height = gray.rows()
        if (width < 200 || height < 200) return null
        val unit = width / 1241.0
        val circles = detectCircles(gray, unit, useSoftMask)
        if (circles.size < 40) return null

        val templateColumns = DoubleArray(COLUMNS) { index ->
            val block = index / 5
            val letter = index % 5
            template.questions.first { it.number == block * 15 + 1 }.options[letter].x
        }
        val templateRows = DoubleArray(ROWS) { row ->
            template.questions.first { it.number == row + 1 }.options[0].y
        }

        val columnPeaks = peaks(circles.map { it.x }, unit)
        val rowPeaks = peaks(circles.map { it.y }, unit)
        trace?.invoke("peaks cols=${columnPeaks.size} rows=${rowPeaks.size}")
        if (trace != null) {
            trace("colPeaks " + columnPeaks.joinToString(" ") { String.format(Locale.US, "%.1f", it) })
            trace("rowPeaks " + rowPeaks.joinToString(" ") { String.format(Locale.US, "%.1f", it) })
        }
        if (columnPeaks.size < 8 || rowPeaks.size < 8) return null

        val rowFit = align(rowPeaks, templateRows, unit) ?: return null
        if (rowFit.matched < 10) return null

        // Column pitch comes from the printed lattice itself: the median gap between
        // detected columns is the within-block pitch, which is unaffected by the wide
        // gaps between blocks and by stray detections elsewhere on the page.
        val columnPitch = medianGap(columnPeaks) ?: return null
        if (columnPitch < 8.0 * unit) return null
        val pitchScale = columnPitch / templatePitch(templateColumns)
        if (!pitchScale.isFinite() || pitchScale < 0.45 || pitchScale > 1.7) return null
        val columnTolerance = max(3.0 * unit, 0.30 * columnPitch)

        val globalOffset = voteOffset(columnPeaks, templateColumns, pitchScale, columnTolerance)
            ?: return null
        trace?.invoke(
            String.format(
                Locale.US,
                "columns pitch=%.1f scale=%.3f offset=%.1f",
                columnPitch,
                pitchScale,
                globalOffset,
            ),
        )

        val columns = DoubleArray(COLUMNS)
        for (block in 0 until BLOCKS) {
            val first = pitchScale * templateColumns[block * PER_BLOCK] + globalOffset
            val last = pitchScale * templateColumns[block * PER_BLOCK + PER_BLOCK - 1] + globalOffset
            val group = columnPeaks.filter { it >= first - 0.75 * columnPitch && it <= last + 0.75 * columnPitch }
            val alignment = alignGroup(
                group,
                templateColumns,
                block,
                pitchScale,
                columnPitch,
                columnTolerance,
                globalOffset,
            )
            val a = alignment?.a ?: pitchScale
            val b = alignment?.b ?: globalOffset
            if (trace != null) {
                trace(
                    String.format(
                        Locale.US,
                        "block %d peaks=%d scale=%.3f offset=%.1f",
                        block,
                        alignment?.matched ?: 0,
                        a,
                        b,
                    ),
                )
            }
            for (index in block * PER_BLOCK until block * PER_BLOCK + PER_BLOCK) {
                columns[index] = a * templateColumns[index] + b
            }
        }
        if (columns.any { !it.isFinite() }) return null

        val matchedColumns = columns.count { column ->
            columnPeaks.any { abs(it - column) <= columnTolerance }
        }
        if (matchedColumns < 10) return null

        val rows = DoubleArray(ROWS) { rowFit.a * templateRows[it] + rowFit.b }
        val pitch = (rows[14] - rows[0]) / 14.0
        if (pitch < 8.0 * unit) return null
        val tolerance = max(3.0 * unit, pitch * 0.35)

        // Anchor the lattice on the printed rings: the initial fit is linear, so a page
        // warp that is slightly off leaves the whole pattern shifted. Two passes of
        // match-then-snap move every column and row onto the median center of the
        // circles matched to its own cells, which is what the reading and the answer
        // box are drawn from.
        var cellMatches = matchCells(circles, template, columns, rows, tolerance)
        var snapped = snapLattice(cellMatches, columns, rows)
        cellMatches = matchCells(circles, template, snapped.first, snapped.second, tolerance)
        snapped = snapLattice(cellMatches, snapped.first, snapped.second)
        val finalColumns = snapped.first
        val finalRows = snapped.second

        val matched = cellMatches.size
        val radiusSum = cellMatches.sumOf { it.circle.r }
        val total = template.questions.sumOf { it.options.size }
        if (matched == 0) return null
        val finalMatchedColumns = finalColumns.count { column ->
            columnPeaks.any { abs(it - column) <= columnTolerance }
        }

        val expected = Array(COLUMNS) { BooleanArray(ROWS) }
        for (question in template.questions) {
            val block = (question.number - 1) / 15
            val row = when {
                question.number <= 15 -> question.number - 1
                question.number <= 30 -> question.number - 16
                else -> question.number - 31
            }
            for (index in 0 until 5) expected[block * 5 + index][row] = true
        }
        var extra = 0
        for (circle in circles) {
            val column = finalColumns.indices.minByOrNull { abs(finalColumns[it] - circle.x) } ?: continue
            val row = finalRows.indices.minByOrNull { abs(finalRows[it] - circle.y) } ?: continue
            if (abs(finalColumns[column] - circle.x) > tolerance || abs(finalRows[row] - circle.y) > tolerance) continue
            if (!expected[column][row]) extra++
        }

        return Fit(
            columns = finalColumns,
            rows = finalRows,
            radius = (radiusSum / matched).coerceIn(3.0, 40.0),
            matchedCells = matched,
            totalCells = total,
            extraCells = extra,
            matchedColumns = finalMatchedColumns,
            matchedRows = rowFit.matched,
        )
    }

    private data class CellMatch(val column: Int, val row: Int, val circle: Circle)

    /** Nearest printed circle within [tolerance] of every template cell. */
    private fun matchCells(
        circles: List<Circle>,
        template: ExamTemplate,
        columns: DoubleArray,
        rows: DoubleArray,
        tolerance: Double,
    ): List<CellMatch> {
        val matches = ArrayList<CellMatch>(columns.size * rows.size)
        for (question in template.questions) {
            val block = (question.number - 1) / 15
            val row = when {
                question.number <= 15 -> question.number - 1
                question.number <= 30 -> question.number - 16
                else -> question.number - 31
            }
            if (row !in rows.indices) continue
            for (option in question.options) {
                val letterIndex = LETTERS.indexOf(option.letter)
                if (letterIndex < 0) continue
                val column = block * 5 + letterIndex
                if (column !in columns.indices) continue
                val x = columns[column]
                val y = rows[row]
                val circle = circles.minByOrNull { (it.x - x) * (it.x - x) + (it.y - y) * (it.y - y) }
                if (circle != null && abs(circle.x - x) <= tolerance && abs(circle.y - y) <= tolerance) {
                    matches.add(CellMatch(column, row, circle))
                }
            }
        }
        return matches
    }

    /**
     * Moves each column and row onto the median center of the circles matched to its
     * cells. Columns and rows with fewer than three matched circles keep the linear
     * fit, so a sparse area cannot drag its line away.
     */
    private fun snapLattice(
        matches: List<CellMatch>,
        columns: DoubleArray,
        rows: DoubleArray,
    ): Pair<DoubleArray, DoubleArray> {
        val snappedColumns = columns.copyOf()
        val snappedRows = rows.copyOf()
        val columnValues = Array(columns.size) { ArrayList<Double>(rows.size) }
        val rowValues = Array(rows.size) { ArrayList<Double>(columns.size) }
        for (match in matches) {
            columnValues[match.column].add(match.circle.x)
            rowValues[match.row].add(match.circle.y)
        }
        for (index in columns.indices) {
            if (columnValues[index].size >= 3) {
                snappedColumns[index] = median(columnValues[index])
            }
        }
        for (index in rows.indices) {
            if (rowValues[index].size >= 3) {
                snappedRows[index] = median(rowValues[index])
            }
        }
        return snappedColumns to snappedRows
    }

    private data class BlockAlignment(val a: Double, val b: Double, val matched: Int)

    /** Average distance between two neighbouring columns inside a block of the template. */
    private fun templatePitch(columns: DoubleArray): Double {
        var sum = 0.0
        var count = 0
        for (block in 0 until BLOCKS) {
            for (index in 1 until PER_BLOCK) {
                sum += columns[block * PER_BLOCK + index] - columns[block * PER_BLOCK + index - 1]
                count++
            }
        }
        return if (count == 0) 1.0 else sum / count
    }

    private fun medianGap(peaks: List<Double>): Double? {
        if (peaks.size < 2) return null
        val gaps = ArrayList<Double>(peaks.size - 1)
        for (index in 1 until peaks.size) gaps.add(peaks[index] - peaks[index - 1])
        gaps.sort()
        return gaps[gaps.size / 2]
    }

    /**
     * Aligns one block of five template columns to the peaks around that block.
     *
     * The scale is fixed (it comes from the printed pitch) and only the offset is searched
     * near the expected position, so a missing or stray peak cannot drag the block by a
     * whole column, which is the failure mode of a plain global least-squares fit.
     */
    private fun alignGroup(
        peaks: List<Double>,
        model: DoubleArray,
        block: Int,
        scale: Double,
        pitch: Double,
        tolerance: Double,
        expectedOffset: Double,
    ): BlockAlignment? {
        if (peaks.size < 2) return null
        val firstIndex = block * PER_BLOCK
        val modelBlock = DoubleArray(PER_BLOCK) { model[firstIndex + it] }
        val offsetLimit = 0.6 * pitch
        var bestOffset = Double.NaN
        var bestCount = 0
        var bestResidual = Double.MAX_VALUE
        for (peak in peaks) {
            for (index in 0 until PER_BLOCK) {
                val offset = peak - scale * modelBlock[index]
                if (abs(offset - expectedOffset) > offsetLimit) continue
                val pairs = matchPairs(peaks, modelBlock, scale, offset, tolerance)
                if (pairs.size < 2) continue
                var residual = 0.0
                for ((modelIndex, peakIndex) in pairs) {
                    val distance = peaks[peakIndex] - (scale * modelBlock[modelIndex] + offset)
                    residual += distance * distance
                }
                if (pairs.size > bestCount || (pairs.size == bestCount && residual < bestResidual)) {
                    bestCount = pairs.size
                    bestOffset = offset
                    bestResidual = residual
                }
            }
        }
        if (bestCount < 2) return null

        val pairs = matchPairs(peaks, modelBlock, scale, bestOffset, tolerance)
        val xs = ArrayList<Double>(pairs.size)
        val us = ArrayList<Double>(pairs.size)
        for ((modelIndex, peakIndex) in pairs) {
            xs.add(modelBlock[modelIndex])
            us.add(peaks[peakIndex])
        }
        val refined = leastSquares(xs, us)
        val a = refined?.first?.takeIf { abs(it - scale) <= 0.12 * scale } ?: scale
        val b = if (refined != null && abs(refined.first - scale) <= 0.12 * scale) {
            refined.second
        } else {
            bestOffset
        }
        return BlockAlignment(a, b, pairs.size)
    }

    /**
     * Finds the lattice offset that places the most template columns on a detected peak.
     * Tries every template column against every peak as the candidate offset.
     */
    private fun voteOffset(
        peaks: List<Double>,
        model: DoubleArray,
        scale: Double,
        tolerance: Double,
    ): Double? {
        var bestOffset = Double.NaN
        var bestCount = 0
        var bestResidual = Double.MAX_VALUE
        for (peak in peaks) {
            for (index in model.indices) {
                val offset = peak - scale * model[index]
                val pairs = matchPairs(peaks, model, scale, offset, tolerance)
                if (pairs.size < bestCount) continue
                var residual = 0.0
                for ((modelIndex, peakIndex) in pairs) {
                    val distance = peaks[peakIndex] - (scale * model[modelIndex] + offset)
                    residual += distance * distance
                }
                if (pairs.size > bestCount || residual < bestResidual) {
                    bestCount = pairs.size
                    bestOffset = offset
                    bestResidual = residual
                }
            }
        }
        return if (bestCount >= 8) bestOffset else null
    }

    /** Pairs template columns with peaks, nearest first, never reusing a peak. */
    private fun matchPairs(
        peaks: List<Double>,
        model: DoubleArray,
        scale: Double,
        offset: Double,
        tolerance: Double,
    ): List<Pair<Int, Int>> {
        val pairs = ArrayList<Pair<Int, Int>>(model.size)
        val used = BooleanArray(peaks.size)
        for (modelIndex in model.indices) {
            val predicted = scale * model[modelIndex] + offset
            var bestIndex = -1
            var bestDistance = tolerance
            for (peakIndex in peaks.indices) {
                if (used[peakIndex]) continue
                val distance = abs(peaks[peakIndex] - predicted)
                if (distance <= bestDistance) {
                    bestDistance = distance
                    bestIndex = peakIndex
                }
            }
            if (bestIndex >= 0) {
                used[bestIndex] = true
                pairs.add(modelIndex to bestIndex)
            }
        }
        return pairs
    }

    fun read(
        gray: Mat,
        template: ExamTemplate,
        fit: Fit,
    ): List<OutcomeAnswer> {
        val height = gray.rows()
        val width = gray.cols()
        val filled = Mat()
        Imgproc.threshold(
            gray,
            filled,
            max(1.0, otsuThreshold(gray) - 18.0),
            255.0,
            Imgproc.THRESH_BINARY_INV,
        )
        val integral = Mat()
        Imgproc.integral(filled, integral, CvType.CV_32S)
        val stride = integral.cols()
        val data = IntArray(integral.rows() * stride)
        integral.get(0, 0, data)
        integral.release()
        filled.release()

        // Small disc: stays clear of the printed ring's ink, so an empty bubble reads
        // near zero while a pen blob fills it almost completely.
        val kernel = discKernel(max(3, (fit.radius * 0.55).roundToInt()))
        val answers = ArrayList<OutcomeAnswer>(template.questions.size)
        for (question in template.questions) {
            val block = (question.number - 1) / 15
            val row = when {
                question.number <= 15 -> question.number - 1
                question.number <= 30 -> question.number - 16
                else -> question.number - 31
            }
            var bestLetter = ""
            var bestRatio = 0.0
            var secondRatio = 0.0
            for (option in question.options) {
                val letterIndex = LETTERS.indexOf(option.letter)
                if (letterIndex < 0) continue
                val cx = fit.columns[block * 5 + letterIndex].roundToInt()
                val cy = fit.rows[row].roundToInt()
                val ratio = discFill(data, stride, width, height, cx, cy, kernel)
                if (ratio > bestRatio) {
                    secondRatio = bestRatio
                    bestRatio = ratio
                    bestLetter = option.letter.toString()
                } else if (ratio > secondRatio) {
                    secondRatio = ratio
                }
            }
            val ambiguous = bestRatio >= 0.42 && secondRatio >= 0.32
            answers.add(
                OutcomeAnswer(
                    number = question.number,
                    letter = if (bestRatio >= 0.42) bestLetter else "",
                    ratio = bestRatio,
                    secondRatio = secondRatio,
                    ambiguous = ambiguous,
                ),
            )
        }
        return answers
    }

    /** Ring darkness of the printed bubbles at the fitted lattice, 0..255. */
    fun ringStrength(gray: Mat, fit: Fit): Double {
        val width = gray.cols()
        val height = gray.rows()
        val radius = max(4, fit.radius.roundToInt())
        val integral = Mat()
        Imgproc.integral(gray, integral, CvType.CV_32S)
        val stride = integral.cols()
        val data = IntArray(integral.rows() * stride)
        integral.get(0, 0, data)
        integral.release()
        val ring = ringKernel(radius)
        val values = ArrayList<Double>(COLUMNS * ROWS)
        for (column in 0 until COLUMNS) {
            for (row in 0 until ROWS) {
                val cx = fit.columns[column].roundToInt()
                val cy = fit.rows[row].roundToInt()
                val mean = boxMean(data, stride, width, height, cx, cy, ring)
                if (mean < 0.0) continue
                values.add(mean)
            }
        }
        if (values.isEmpty()) return 0.0
        return 255.0 - median(values)
    }

    private data class Alignment(val a: Double, val b: Double, val matched: Int)

    /**
     * Fits detected peak positions onto the template lattice positions, trying every pair
     * of peaks as an initial guess so stray detections cannot drag the fit away.
     */
    private fun align(peaks: List<Double>, model: DoubleArray, unit: Double): Alignment? {
        val modelFirst = model.first()
        val modelSpan = model.last() - modelFirst
        var best: Alignment? = null
        for (first in peaks.indices) {
            for (second in first + 1 until peaks.size) {
                val span = peaks[second] - peaks[first]
                if (span < 40.0 * unit) continue
                var a = span / modelSpan
                if (a < 0.5 || a > 1.6) continue
                var b = peaks[first] - a * modelFirst
                var matched = 0
                var accepted = false
                for (iteration in 0 until 3) {
                    val tolerance = if (iteration == 0) 16.0 * unit else 8.0 * unit
                    val xs = ArrayList<Double>(model.size)
                    val us = ArrayList<Double>(model.size)
                    val used = BooleanArray(peaks.size)
                    for (value in model) {
                        val predicted = a * value + b
                        var bestIndex = -1
                        var bestDistance = tolerance
                        for (index in peaks.indices) {
                            if (used[index]) continue
                            val distance = abs(peaks[index] - predicted)
                            if (distance <= bestDistance) {
                                bestDistance = distance
                                bestIndex = index
                            }
                        }
                        if (bestIndex >= 0) {
                            used[bestIndex] = true
                            xs.add(value)
                            us.add(peaks[bestIndex])
                        }
                    }
                    if (xs.size < 6) {
                        accepted = false
                        break
                    }
                    val fit = leastSquares(xs, us)
                    if (fit == null || fit.first < 0.5 || fit.first > 1.6) {
                        accepted = false
                        break
                    }
                    a = fit.first
                    b = fit.second
                    matched = xs.size
                    accepted = true
                }
                if (accepted && (best == null || matched > best.matched)) {
                    best = Alignment(a, b, matched)
                }
            }
        }
        return best
    }

    private fun leastSquares(xs: List<Double>, ys: List<Double>): Pair<Double, Double>? {
        val n = xs.size.toDouble()
        if (n < 2) return null
        val meanX = xs.sum() / n
        val meanY = ys.sum() / n
        var sxx = 0.0
        var sxy = 0.0
        for (index in xs.indices) {
            val x = xs[index] - meanX
            sxx += x * x
            sxy += x * (ys[index] - meanY)
        }
        if (sxx < 1e-6) return null
        val a = sxy / sxx
        if (!a.isFinite()) return null
        return a to (meanY - a * meanX)
    }

    private fun detectCircles(gray: Mat, unit: Double, useSoftMask: Boolean = false): List<Circle> {
        val improved = Mat()
        Imgproc.createCLAHE(2.0, Size(8.0, 8.0)).apply(gray, improved)
        val blurred = Mat()
        Imgproc.GaussianBlur(improved, blurred, Size(3.0, 3.0), 0.0)
        val masks = ArrayList<Pair<Mat, Double>>(3)
        try {
            val adaptiveBlock = max(15, (31 * unit).roundToInt() or 1)
            val adaptive = Mat()
            Imgproc.adaptiveThreshold(
                blurred,
                adaptive,
                255.0,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV,
                adaptiveBlock,
                max(3.0, 6.0 * unit),
            )
            masks.add(adaptive to 0.78)

            val kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                Size(max(9.0, 25 * unit), max(9.0, 25 * unit)),
            )
            val blackHat = Mat()
            Imgproc.morphologyEx(blurred, blackHat, Imgproc.MORPH_BLACKHAT, kernel)
            kernel.release()
            val blackHatMask = Mat()
            Imgproc.threshold(
                blackHat,
                blackHatMask,
                0.0,
                255.0,
                Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU,
            )
            blackHat.release()
            masks.add(blackHatMask to 0.78)

            // Rescue mask for thin prints and binarized scans: a strong blur turns
            // their thin rings into soft dark disks. Relaxed circularity, because the
            // soft disks are less round than crisp printed rings.
            if (useSoftMask) {
                val soft = Mat()
                Imgproc.GaussianBlur(blurred, soft, Size(9.0, 9.0), 0.0)
                val softMask = Mat()
                Imgproc.adaptiveThreshold(
                    soft,
                    softMask,
                    255.0,
                    Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                    Imgproc.THRESH_BINARY_INV,
                    adaptiveBlock,
                    max(5.0, 9.0 * unit),
                )
                soft.release()
                masks.add(softMask to 0.55)
            }

            val raw = ArrayList<Circle>(256)
            val minArea = 60.0 * unit * unit
            val maxArea = 1600.0 * unit * unit
            val minRadius = 5.0 * unit
            val maxRadius = 22.0 * unit
            for ((mask, minCircularity) in masks) {
                val contours = ArrayList<MatOfPoint>()
                val hierarchy = Mat()
                Imgproc.findContours(
                    mask,
                    contours,
                    hierarchy,
                    Imgproc.RETR_LIST,
                    Imgproc.CHAIN_APPROX_SIMPLE,
                )
                hierarchy.release()
                for (contour in contours) {
                    val area = Imgproc.contourArea(contour)
                    if (area < minArea || area > maxArea) {
                        contour.release()
                        continue
                    }
                    val curve = MatOfPoint2f(*contour.toArray())
                    contour.release()
                    val perimeter = Imgproc.arcLength(curve, true)
                    val center = Point()
                    val radius = FloatArray(1)
                    Imgproc.minEnclosingCircle(curve, center, radius)
                    curve.release()
                    if (perimeter <= 0.0) continue
                    val circularity = 4.0 * PI * area / (perimeter * perimeter)
                    if (circularity < minCircularity) continue
                    if (radius[0] < minRadius || radius[0] > maxRadius) continue
                    raw.add(Circle(center.x, center.y, radius[0].toDouble()))
                }
            }

            // The inner and outer boundary of one printed ring both look circular.
            val kept = ArrayList<Circle>(raw.size)
            for (circle in raw.sortedByDescending { it.r }) {
                if (kept.none { abs(it.x - circle.x) <= 4.0 * unit && abs(it.y - circle.y) <= 4.0 * unit }) {
                    kept.add(circle)
                }
            }
            return kept
        } finally {
            masks.forEach { (mask, _) -> mask.release() }
            blurred.release()
            improved.release()
        }
    }

    private fun peaks(values: List<Double>, unit: Double): List<Double> {
        if (values.isEmpty()) return emptyList()
        val low = values.min()
        val size = (values.max() - low).toInt() + 1
        val histogram = DoubleArray(size)
        for (value in values) {
            histogram[(value - low).toInt().coerceIn(0, size - 1)] += 1.0
        }
        val half = max(1, (2 * unit).roundToInt())
        val smooth = DoubleArray(size) { index ->
            (max(0, index - half)..minOf(size - 1, index + half)).sumOf { histogram[it] }
        }
        val threshold = max(3.0, 0.08 * (smooth.maxOrNull() ?: 0.0))
        val found = ArrayList<Double>(20)
        var index = 1
        while (index < size - 1) {
            val value = smooth[index]
            if (value >= threshold && value >= smooth[index - 1] && value >= smooth[index + 1]) {
                var end = index
                while (end + 1 < size && smooth[end + 1] == value) end++
                val center = low + (index + end) / 2.0
                if (found.isNotEmpty() && center - found.last() < 10.0 * unit) {
                    found[found.size - 1] = (found.last() + center) / 2.0
                } else {
                    found.add(center)
                }
                index = end + 1
            } else {
                index++
            }
        }
        return found
    }

    private fun otsuThreshold(gray: Mat): Double {
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        val binary = Mat()
        val value = Imgproc.threshold(blurred, binary, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        blurred.release()
        binary.release()
        return value
    }

    private data class KernelRow(val dy: Int, val segments: List<IntArray>)
    private data class Kernel(val rows: List<KernelRow>, val area: Int)

    private fun discKernel(radius: Int): Kernel {
        val rows = ArrayList<KernelRow>(2 * radius + 1)
        var area = 0
        for (dy in -radius..radius) {
            val reach = sqrt((radius.toDouble() * radius - dy.toDouble() * dy).coerceAtLeast(0.0)).toInt()
            rows.add(KernelRow(dy, listOf(intArrayOf(-reach, reach))))
            area += 2 * reach + 1
        }
        return Kernel(rows, max(1, area))
    }

    private fun ringKernel(radius: Int): Kernel {
        val inner = max(1, radius - 3)
        val rows = ArrayList<KernelRow>(2 * radius + 1)
        var area = 0
        for (dy in -radius..radius) {
            val outer = sqrt((radius.toDouble() * radius - dy.toDouble() * dy).coerceAtLeast(0.0)).toInt()
            if (outer == 0) continue
            val hole = if (abs(dy) < inner) {
                sqrt((inner.toDouble() * inner - dy.toDouble() * dy).coerceAtLeast(0.0)).toInt()
            } else {
                0
            }
            val segments = if (hole <= 0) {
                listOf(intArrayOf(-outer, outer))
            } else {
                listOf(intArrayOf(-outer, -hole), intArrayOf(hole, outer))
            }
            rows.add(KernelRow(dy, segments))
            for (segment in segments) area += segment[1] - segment[0] + 1
        }
        return Kernel(rows, max(1, area))
    }

    private fun discFill(
        data: IntArray,
        stride: Int,
        width: Int,
        height: Int,
        centerX: Int,
        centerY: Int,
        kernel: Kernel,
    ): Double {
        val mean = boxMean(data, stride, width, height, centerX, centerY, kernel)
        if (mean < 0.0) return 0.0
        return mean / 255.0
    }

    private fun boxMean(
        data: IntArray,
        stride: Int,
        width: Int,
        height: Int,
        centerX: Int,
        centerY: Int,
        kernel: Kernel,
    ): Double {
        var sum = 0L
        for (row in kernel.rows) {
            val y = centerY + row.dy
            if (y < 0 || y >= height) return -1.0
            for (segment in row.segments) {
                val x0 = centerX + segment[0]
                val x1 = centerX + segment[1]
                if (x0 < 0 || x1 >= width) return -1.0
                val top = y * stride
                val bottom = (y + 1) * stride
                sum += data[bottom + x1 + 1] - data[bottom + x0] -
                    data[top + x1 + 1] + data[top + x0]
            }
        }
        return sum.toDouble() / kernel.area
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        }
    }
}
