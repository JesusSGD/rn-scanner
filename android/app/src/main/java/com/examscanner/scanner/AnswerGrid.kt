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
 *
 * Every sheet of the family prints the same five-option block (A-E by fifteen rows), but
 * the exam may need one to six blocks side by side, and the last block may print only
 * some of its rows. Both come from the page itself, so a sheet with two blocks fits a
 * two-block grid instead of being forced onto the three-block template.
 */
object AnswerGrid {
    const val PER_BLOCK = 5
    const val LETTERS = "ABCDE"
    const val QUESTIONS_PER_BLOCK = 15
    private const val MAX_BLOCKS = 6
    // A pixel this much darker than the paper beside it is ink. The level is measured per
    // question row, because exposure and shadows drift across a photographed page.
    private const val INK_MARGIN = 0.10
    private const val MIN_INK_MARGIN = 12.0

    data class Fit(
        /** One x coordinate per option column: block * [PER_BLOCK] + letter. */
        val columns: DoubleArray,
        /** One y coordinate per printed row of a block. */
        val rows: DoubleArray,
        /** Printed rows per block; the last block may hold fewer than the others. */
        val blockRows: IntArray,
        val radius: Double,
        val matchedCells: Int,
        val extraCells: Int,
        val matchedColumns: Int,
        val matchedRows: Int,
        /**
         * Distance between the fitted rows and where the template says the answer grid
         * sits on the page, as a fraction of the page height. The grid always lies in the
         * lower half of the sheet, which is what tells a sheet read upside down from the
         * right side when the printed layout itself is vertically symmetric (one block,
         * or every block complete).
         */
        val positionError: Double,
    ) {
        /** Cells of the detected structure only; blank paper after the last block is excluded. */
        val totalCells: Int get() = PER_BLOCK * blockRows.sum()

        /** Fraction of expected bubbles that sit on a detected printed ring. */
        val coverage: Double get() = if (totalCells == 0) 0.0 else matchedCells.toDouble() / totalCells

        /**
         * Score used to pick the page orientation: real cells count, foreign rings do not,
         * and a lattice away from the grid's usual place on the page is penalised.
         */
        val latticeScore: Double
            get() = matchedCells.toDouble() - 0.5 * extraCells +
                0.5 * (matchedColumns + matchedRows) - 100.0 * positionError

        /** Maps a fit computed on a downscaled page warp onto the full-scale warp. */
        fun scaled(factor: Double): Fit = copy(
            columns = DoubleArray(columns.size) { columns[it] * factor },
            rows = DoubleArray(rows.size) { rows[it] * factor },
            radius = radius * factor,
            blockRows = blockRows.copyOf(),
        )
    }

    private data class Circle(val x: Double, val y: Double, val r: Double)

    /**
     * Fixed geometry of the sheet family, recovered from the template: where each block
     * starts, how the five option columns are spaced inside a block, and where the rows
     * sit. Only the template's own blocks are known; the remaining ones are extrapolated
     * so sheets with four to six blocks can still be anchored.
     */
    private data class Model(
        val blockOrigins: DoubleArray,
        val columnOffsets: DoubleArray,
        val templateRows: DoubleArray,
    ) {
        /** Absolute x of every option column of the six possible blocks. */
        fun columns(): DoubleArray = DoubleArray(MAX_BLOCKS * PER_BLOCK) { index ->
            blockOrigins[index / PER_BLOCK] + columnOffsets[index % PER_BLOCK]
        }
    }

    private fun buildModel(template: ExamTemplate): Model? {
        if (template.questions.isEmpty()) return null
        val rowsByBlock = HashMap<Int, MutableList<Double>>()
        val firstColumnByBlock = HashMap<Int, Double>()
        for (question in template.questions) {
            val block = (question.number - 1) / QUESTIONS_PER_BLOCK
            rowsByBlock.getOrPut(block) { ArrayList() }.add(question.options.first().y)
            val left = question.options.minOf { it.x }
            firstColumnByBlock[block] = minOf(firstColumnByBlock[block] ?: Double.MAX_VALUE, left)
        }

        // The block with the most printed rows gives the row model (normally the first).
        val templateRows = rowsByBlock.values.maxByOrNull { it.size }?.sorted()?.toDoubleArray()
            ?: return null
        if (templateRows.size < 4) return null

        val knownBlocks = firstColumnByBlock.keys.sorted()
        if (knownBlocks.isEmpty()) return null
        val origins = ArrayList<Double>(MAX_BLOCKS)
        for (block in knownBlocks) origins.add(firstColumnByBlock.getValue(block))
        val pitch = if (origins.size >= 2) (origins.last() - origins.first()) / (origins.size - 1) else 0.0
        while (origins.size < MAX_BLOCKS) {
            if (pitch <= 0.0) break
            origins.add(origins.last() + pitch)
        }

        val firstQuestion = template.questions.first {
            (it.number - 1) / QUESTIONS_PER_BLOCK == knownBlocks.first()
        }
        val optionXs = firstQuestion.options.map { it.x }.sorted()
        if (optionXs.size != PER_BLOCK) return null
        val offsets = DoubleArray(PER_BLOCK) { optionXs[it] - optionXs[0] }

        return Model(origins.toDoubleArray(), offsets, templateRows)
    }

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
        val model = buildModel(template) ?: return null
        val modelColumns = model.columns()

        val integral = Mat()
        Imgproc.integral(gray, integral, CvType.CV_32S)
        val stride = integral.cols()
        val data = IntArray(integral.rows() * stride)
        integral.get(0, 0, data)
        integral.release()
        val ringRadius = max(5, ((template.bubbleRadius + 1.0) * unit).roundToInt())
        val ring = ringKernel(ringRadius)
        val center = discKernel(3)

        fun contrast(cx: Int, cy: Int): Boolean {
            val ringMean = boxMean(data, stride, width, height, cx, cy, ring)
            val centerMean = boxMean(data, stride, width, height, cx, cy, center)
            // A printed bubble reads as paper inside a dark ring; text or stray
            // marks are dark through and through, so this contrast only holds
            // for real rings.
            return centerMean > ringMean + 4.0
        }

        fun evaluate(scaleX: Double, scaleY: Double, dx: Double, dy: Double): Triple<Int, DoubleArray, DoubleArray> {
            val columns = DoubleArray(modelColumns.size) { scaleX * modelColumns[it] + dx }
            val rows = DoubleArray(model.templateRows.size) { scaleY * model.templateRows[it] + dy }
            var rings = 0
            for (column in columns) {
                val cx = column.roundToInt()
                for (row in rows) {
                    if (contrast(cx, row.roundToInt())) rings++
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

        val columns = best.second
        val rows = best.third
        val rowCount = rows.size

        // Which blocks and rows are actually printed: a row counts as printed when at
        // least three of its five bubbles show the ring contrast. Blocks are taken as a
        // contiguous run from the first one, and a printed block always has its top
        // rows, so the first block without printed rows ends the structure.
        val blockRows = IntArray(MAX_BLOCKS)
        for (block in 0 until MAX_BLOCKS) {
            if (block > 0 && blockRows[block - 1] == 0) break
            var topPrinted = false
            var last = -1
            for (row in 0 until rowCount) {
                var printed = 0
                for (letter in 0 until PER_BLOCK) {
                    if (contrast(columns[block * PER_BLOCK + letter].roundToInt(), rows[row].roundToInt())) printed++
                }
                if (printed >= 3) {
                    if (row == 0) topPrinted = true
                    last = row
                }
            }
            blockRows[block] = if (topPrinted) last + 1 else 0
        }
        var blockCount = blockRows.size
        while (blockCount > 0 && blockRows[blockCount - 1] == 0) blockCount--
        if (blockCount == 0) {
            trace?.invoke("template fit rings=${best.first} blocks=0")
            return null
        }
        val structure = blockRows.copyOf(blockCount)
        val finalColumns = columns.copyOf(blockCount * PER_BLOCK)
        val total = PER_BLOCK * structure.sum()

        // Every ring found inside the structure is a matched cell; a ring-like blob
        // outside it (below the last block's rows, or where no block is printed) means
        // the template is upside down or shifted, so it is evidence against this fit.
        val columnRings = IntArray(blockCount * PER_BLOCK)
        val rowRings = IntArray(rowCount)
        for (block in 0 until blockCount) {
            for (row in 0 until structure[block]) {
                for (letter in 0 until PER_BLOCK) {
                    if (contrast(columns[block * PER_BLOCK + letter].roundToInt(), rows[row].roundToInt())) {
                        columnRings[block * PER_BLOCK + letter]++
                        rowRings[row]++
                    }
                }
            }
        }
        var blankHits = 0
        for (block in 0 until MAX_BLOCKS) {
            for (row in 0 until rowCount) {
                if (block < blockCount && row < structure[block]) continue
                for (letter in 0 until PER_BLOCK) {
                    if (contrast(columns[block * PER_BLOCK + letter].roundToInt(), rows[row].roundToInt())) blankHits++
                }
            }
        }
        val rings = columnRings.sum()
        if (rings < total * 0.5) {
            trace?.invoke("template fit rings=$rings blocks=$blockCount")
            return null
        }
        trace?.invoke(
            "template fit rings=$rings blank=$blankHits blocks=$blockCount rows=${structure.joinToString(",")} " +
                "scale=$bestScaleX/$bestScaleY offset=${bestOffsetX},${bestOffsetY}",
        )
        var matchedRows = 0
        for (row in 0 until rowCount) {
            var rowCells = 0
            for (block in 0 until blockCount) if (row < structure[block]) rowCells += PER_BLOCK
            if (rowCells > 0 && rowRings[row] >= (rowCells * 3) / 4) matchedRows++
        }
        return Fit(
            columns = finalColumns,
            rows = rows,
            blockRows = structure,
            radius = ringRadius.toDouble(),
            matchedCells = rings,
            extraCells = 2 * blankHits,
            matchedColumns = columnRings.indices.count { columnRings[it] >= (structure[it / PER_BLOCK] * 3) / 4 },
            matchedRows = matchedRows,
            positionError = positionError(model, unit, rows, height),
        )
    }

    private fun fitFromCircles(
        gray: Mat,
        template: ExamTemplate,
        trace: ((String) -> Unit)?,
        useSoftMask: Boolean,
    ): Fit? {
        val width = gray.cols()
        val height = gray.rows()
        if (width < 200 || height < 200) return null
        val unit = width / 1241.0
        val circles = detectCircles(gray, unit, useSoftMask)
        if (circles.size < 40) return null
        val model = buildModel(template) ?: return null

        val columnPeaks = peaks(circles.map { it.x }, unit)
        val rowPeaks = peaks(circles.map { it.y }, unit)
        trace?.invoke("peaks cols=${columnPeaks.size} rows=${rowPeaks.size}")
        if (trace != null) {
            trace("colPeaks " + columnPeaks.joinToString(" ") { String.format(Locale.US, "%.1f", it) })
            trace("rowPeaks " + rowPeaks.joinToString(" ") { String.format(Locale.US, "%.1f", it) })
        }
        if (columnPeaks.size < 4 || rowPeaks.size < 8) return null

        val rowFit = align(rowPeaks, model.templateRows, unit) ?: return null
        if (rowFit.matched < 10) return null
        val rows = DoubleArray(model.templateRows.size) { rowFit.a * model.templateRows[it] + rowFit.b }
        val rowPitch = rowPitch(rows)
        if (rowPitch < 8.0 * unit) return null
        val cellTolerance = max(3.0 * unit, rowPitch * 0.35)

        // Column pitch comes from the printed lattice itself: the median gap between
        // detected columns is the within-block pitch, which is unaffected by the wide
        // gaps between blocks and by stray detections elsewhere on the page.
        val columnPitch = medianGap(columnPeaks) ?: return null
        if (columnPitch < 8.0 * unit) return null
        val pitchScale = columnPitch / offsetsPitch(model.columnOffsets)
        if (!pitchScale.isFinite() || pitchScale < 0.45 || pitchScale > 1.7) return null
        val columnTolerance = max(3.0 * unit, 0.30 * columnPitch)

        var blocks = detectBlocks(
            model,
            columnPeaks,
            circles,
            rows,
            cellTolerance,
            pitchScale,
            columnPitch,
            columnTolerance,
            trace,
        ) ?: return null
        var columns = DoubleArray(blocks.size * PER_BLOCK) { index ->
            blocks[index / PER_BLOCK][index % PER_BLOCK]
        }
        if (columns.any { !it.isFinite() }) return null

        // Rows per block: the last block may print only some of its fifteen rows.
        var blockRows = IntArray(blocks.size) { rows.size }
        var cellMatches = matchCells(circles, columns, rows, blockRows, cellTolerance)
        blockRows = trimRows(cellMatches, blocks.size, rows.size)
        // Blocks are printed left to right from the first one; the first block without
        // printed rows ends the structure.
        var blockCount = blockRows.indexOfFirst { it == 0 }.let { if (it < 0) blockRows.size else it }
        if (blockCount == 0) return null
        if (blockCount < blockRows.size) {
            blockRows = blockRows.copyOf(blockCount)
            columns = columns.copyOf(blockCount * PER_BLOCK)
        }
        cellMatches = matchCells(circles, columns, rows, blockRows, cellTolerance)
        var snapped = snapLattice(cellMatches, columns, rows)
        cellMatches = matchCells(circles, snapped.first, snapped.second, blockRows, cellTolerance)
        snapped = snapLattice(cellMatches, snapped.first, snapped.second)
        val finalColumns = snapped.first
        val finalRows = snapped.second

        val matched = cellMatches.size
        if (matched == 0) return null
        val radiusSum = cellMatches.sumOf { it.circle.r }
        val matchedColumns = finalColumns.indices.count { column ->
            cellMatches.count { it.column == column } >= 2
        }
        val finalMatchedColumnsByPeak = finalColumns.count { column ->
            columnPeaks.any { abs(it - column) <= columnTolerance }
        }

        // Rings that sit in the rows band but outside the detected structure are foreign
        // grid evidence: the partial block of an upside-down sheet, a structure shifted
        // by a whole block, or the blank tail of a misdetected lattice. They are what
        // tells a two-block sheet from a three-block one seen from the wrong side.
        val bandTop = finalRows.min() - cellTolerance
        val bandBottom = finalRows.max() + cellTolerance
        var extra = 0
        for (circle in circles) {
            if (circle.y < bandTop || circle.y > bandBottom) continue
            val column = finalColumns.indices.minByOrNull { abs(finalColumns[it] - circle.x) } ?: continue
            val row = finalRows.indices.minByOrNull { abs(finalRows[it] - circle.y) } ?: continue
            val onColumn = abs(finalColumns[column] - circle.x) <= cellTolerance
            val onRow = abs(finalRows[row] - circle.y) <= cellTolerance
            val explained = onColumn && onRow && row < blockRows[column / PER_BLOCK]
            if (!explained) extra++
        }

        trace?.invoke(
            String.format(
                Locale.US,
                "circles blocks=%d rows=%s matched=%d peaks=%d columns=%d",
                blockCount,
                blockRows.joinToString(","),
                matched,
                finalMatchedColumnsByPeak,
                matchedColumns,
            ),
        )
        return Fit(
            columns = finalColumns,
            rows = finalRows,
            blockRows = blockRows,
            radius = (radiusSum / matched).coerceIn(3.0, 40.0),
            matchedCells = matched,
            extraCells = extra,
            matchedColumns = matchedColumns,
            matchedRows = rowFit.matched,
            positionError = positionError(model, unit, finalRows, height),
        )
    }

    /**
     * Finds the printed blocks, left to right, from the detected column peaks.
     *
     * A global offset places the template lattice on the peaks; every candidate block is
     * then aligned inside that offset and accepted while it holds printed rows. A block
     * whose peaks were not detected (a last block with only a few rows) still counts when
     * its bubbles are there, which is why the linear template columns are checked as a
     * fallback. Only when the template anchor finds nothing (a sheet laid out elsewhere)
     * are the peaks split into runs separated by wide gaps and aligned on their own.
     */
    private fun detectBlocks(
        model: Model,
        columnPeaks: List<Double>,
        circles: List<Circle>,
        rows: DoubleArray,
        cellTolerance: Double,
        pitchScale: Double,
        columnPitch: Double,
        peakTolerance: Double,
        trace: ((String) -> Unit)?,
    ): List<DoubleArray>? {
        val modelColumns = model.columns()
        val globalOffset = voteOffset(columnPeaks, modelColumns, pitchScale, peakTolerance)
        if (globalOffset != null) {
            anchoredBlocks(model, columnPeaks, circles, rows, cellTolerance, pitchScale, columnPitch, peakTolerance, globalOffset, trace)
                ?.let { return it }
        }

        // Rescue: split the peaks into runs separated by wide gaps and align each run.
        val groups = ArrayList<MutableList<Double>>()
        for (peak in columnPeaks) {
            val group = groups.lastOrNull()
            if (group != null && peak - group.last() <= 1.7 * columnPitch) {
                group.add(peak)
            } else {
                groups.add(mutableListOf(peak))
            }
        }
        val rescued = ArrayList<DoubleArray>(MAX_BLOCKS)
        for (group in groups) {
            if (group.size < 4) continue
            val alignment = alignGroup(group, model.columnOffsets, pitchScale, columnPitch, peakTolerance, null)
                ?: continue
            val candidate = DoubleArray(PER_BLOCK) { alignment.a * model.columnOffsets[it] + alignment.b }
            if (blockRowPresence(circles, candidate, rows, cellTolerance) == 0) continue
            rescued.add(candidate)
        }
        rescued.sortBy { it.first() }
        if (rescued.isEmpty() || rescued.size > MAX_BLOCKS) return null

        // Anchor the rest of the sheet on the leftmost rescued block: the template offset
        // is known once block one is placed, which recovers blocks whose peaks were lost.
        val anchored = anchoredBlocks(
            model,
            columnPeaks,
            circles,
            rows,
            cellTolerance,
            pitchScale,
            columnPitch,
            peakTolerance,
            rescued.first().first() - pitchScale * model.blockOrigins[0],
            trace,
        )
        return if (anchored != null && anchored.size >= rescued.size) anchored else rescued
    }

    private fun anchoredBlocks(
        model: Model,
        columnPeaks: List<Double>,
        circles: List<Circle>,
        rows: DoubleArray,
        cellTolerance: Double,
        pitchScale: Double,
        columnPitch: Double,
        peakTolerance: Double,
        globalOffset: Double,
        trace: ((String) -> Unit)?,
    ): List<DoubleArray>? {
        val blocks = ArrayList<DoubleArray>(MAX_BLOCKS)
        val span = pitchScale * model.columnOffsets.last()
        for (block in 0 until MAX_BLOCKS) {
            val expected = pitchScale * model.blockOrigins[block] + globalOffset
            val group = columnPeaks.filter {
                it >= expected - 0.75 * columnPitch && it <= expected + span + 0.75 * columnPitch
            }
            val alignment = alignGroup(group, model.columnOffsets, pitchScale, columnPitch, peakTolerance, expected)
            val alignedCandidate = alignment?.let { aligned ->
                DoubleArray(PER_BLOCK) { aligned.a * model.columnOffsets[it] + aligned.b }
            }
            // The peaks of this block were lost (a last block with few printed rows):
            // fall back to the template columns and let the snap pass refine them.
            val candidate = alignedCandidate
                ?.takeIf { blockRowPresence(circles, it, rows, cellTolerance) > 0 }
                ?: DoubleArray(PER_BLOCK) {
                    pitchScale * (model.blockOrigins[block] + model.columnOffsets[it]) + globalOffset
                }
            val presentRows = blockRowPresence(circles, candidate, rows, cellTolerance)
            if (trace != null) {
                trace(
                    String.format(
                        Locale.US,
                        "block %d peaks=%d rows=%d offset=%.1f",
                        block,
                        alignment?.matched ?: 0,
                        presentRows,
                        candidate[0],
                    ),
                )
            }
            if (presentRows == 0) break
            blocks.add(candidate)
        }
        return blocks.ifEmpty { null }
    }

    private data class CellMatch(val column: Int, val row: Int, val circle: Circle)

    /** Nearest printed circle within [tolerance] of every expected cell. */
    private fun matchCells(
        circles: List<Circle>,
        columns: DoubleArray,
        rows: DoubleArray,
        blockRows: IntArray,
        tolerance: Double,
    ): List<CellMatch> {
        val matches = ArrayList<CellMatch>(columns.size * rows.size)
        for (block in blockRows.indices) {
            for (row in 0 until blockRows[block].coerceAtMost(rows.size)) {
                for (letter in 0 until PER_BLOCK) {
                    val column = block * PER_BLOCK + letter
                    val x = columns[column]
                    val y = rows[row]
                    val circle = circles.minByOrNull { (it.x - x) * (it.x - x) + (it.y - y) * (it.y - y) }
                    if (circle != null && abs(circle.x - x) <= tolerance && abs(circle.y - y) <= tolerance) {
                        matches.add(CellMatch(column, row, circle))
                    }
                }
            }
        }
        return matches
    }

    /**
     * Rows printed per block. A row counts as printed when at least three of its five
     * columns carry a detected ring. Every block prints its rows from the top, so a block
     * whose first row is blank does not belong to the structure (it is the partial block
     * seen upside down). Taking the last printed row instead of the first gap keeps the
     * questions of a row whose rings the detector missed.
     */
    private fun trimRows(matches: List<CellMatch>, blockCount: Int, rowCount: Int): IntArray {
        val counts = Array(blockCount) { IntArray(rowCount) }
        for (match in matches) {
            if (match.row in 0 until rowCount) counts[match.column / PER_BLOCK][match.row]++
        }
        return IntArray(blockCount) { block ->
            var last = -1
            for (row in 0 until rowCount) if (counts[block][row] >= 3) last = row
            if (counts[block][0] < 3) 0 else last + 1
        }
    }

    /** Number of rows of [columns] that carry at least three printed bubbles. */
    private fun blockRowPresence(
        circles: List<Circle>,
        columns: DoubleArray,
        rows: DoubleArray,
        tolerance: Double,
    ): Int {
        var present = 0
        for (rowY in rows) {
            var matched = 0
            for (x in columns) {
                val circle = circles.minByOrNull { (it.x - x) * (it.x - x) + (it.y - rowY) * (it.y - rowY) }
                if (circle != null && abs(circle.x - x) <= tolerance && abs(circle.y - rowY) <= tolerance) {
                    matched++
                }
            }
            if (matched >= 3) present++
        }
        return present
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
    private fun offsetsPitch(offsets: DoubleArray): Double =
        if (offsets.size < 2) 1.0 else (offsets.last() - offsets.first()) / (offsets.size - 1)

    private fun rowPitch(rows: DoubleArray): Double =
        if (rows.size < 2) 0.0 else (rows.last() - rows.first()) / (rows.size - 1)

    /**
     * Vertical distance between the fitted rows and the page position the template
     * expects, as a fraction of the page height. Used only to break orientation ties:
     * the answer grid always sits below the instructions, never in the upper half.
     */
    private fun positionError(model: Model, unit: Double, rows: DoubleArray, height: Int): Double {
        if (rows.isEmpty() || height <= 0) return 0.0
        val expected = unit * (model.templateRows.first() + model.templateRows.last()) / 2.0
        val actual = (rows.first() + rows.last()) / 2.0
        return abs(actual - expected) / height
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
     * The scale is fixed (it comes from the printed pitch) and only the offset is searched,
     * so a missing or stray peak cannot drag the block by a whole column, which is the
     * failure mode of a plain global least-squares fit.
     */
    private fun alignGroup(
        peaks: List<Double>,
        offsets: DoubleArray,
        scale: Double,
        pitch: Double,
        tolerance: Double,
        expectedOffset: Double?,
    ): BlockAlignment? {
        if (peaks.size < 2) return null
        val offsetLimit = 0.6 * pitch
        var bestOffset = Double.NaN
        var bestCount = 0
        var bestResidual = Double.MAX_VALUE
        for (peak in peaks) {
            for (index in 0 until PER_BLOCK) {
                val offset = peak - scale * offsets[index]
                if (expectedOffset != null && abs(offset - expectedOffset) > offsetLimit) continue
                val pairs = matchPairs(peaks, offsets, scale, offset, tolerance)
                if (pairs.size < 2) continue
                var residual = 0.0
                for ((modelIndex, peakIndex) in pairs) {
                    val distance = peaks[peakIndex] - (scale * offsets[modelIndex] + offset)
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

        val pairs = matchPairs(peaks, offsets, scale, bestOffset, tolerance)
        val xs = ArrayList<Double>(pairs.size)
        val us = ArrayList<Double>(pairs.size)
        for ((modelIndex, peakIndex) in pairs) {
            xs.add(offsets[modelIndex])
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
        fit: Fit,
    ): List<OutcomeAnswer> {
        val height = gray.rows()
        val width = gray.cols()
        val pixels = ByteArray(width * height)
        gray.get(0, 0, pixels)

        // Small disc: stays clear of the printed ring's ink, so an empty bubble reads
        // near zero while a pen mark fills a good part of it.
        val kernel = discKernel(max(3, (fit.radius * 0.55).roundToInt()))
        val answers = ArrayList<OutcomeAnswer>(fit.totalCells / PER_BLOCK)
        for (block in fit.blockRows.indices) {
            for (row in 0 until fit.blockRows[block]) {
                val cy = fit.rows[row].roundToInt()
                val centers = IntArray(PER_BLOCK) { letter ->
                    fit.columns[block * PER_BLOCK + letter].roundToInt()
                }
                // The five bubbles of a question share a row, so their typical tone is the
                // paper level right there. A soft shadow or a darker corner of the page
                // moves that baseline with them, which one global threshold cannot do.
                val paper = median(
                    centers.map { discMean(pixels, width, height, it, cy, kernel) }.filter { it >= 0.0 },
                )
                val inkThreshold = paper - max(MIN_INK_MARGIN, INK_MARGIN * paper)
                var bestLetter = ""
                var bestRatio = 0.0
                var secondRatio = 0.0
                for (letter in 0 until PER_BLOCK) {
                    val ratio = discInk(pixels, width, height, centers[letter], cy, inkThreshold, kernel)
                    if (ratio > bestRatio) {
                        secondRatio = bestRatio
                        bestRatio = ratio
                        bestLetter = LETTERS[letter].toString()
                    } else if (ratio > secondRatio) {
                        secondRatio = ratio
                    }
                }
                val ambiguous = bestRatio >= 0.42 && secondRatio >= 0.32
                answers.add(
                    OutcomeAnswer(
                        number = block * QUESTIONS_PER_BLOCK + row + 1,
                        letter = if (bestRatio >= 0.42) bestLetter else "",
                        ratio = bestRatio,
                        secondRatio = secondRatio,
                        ambiguous = ambiguous,
                    ),
                )
            }
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
        val values = ArrayList<Double>(fit.totalCells)
        for (block in fit.blockRows.indices) {
            for (row in 0 until fit.blockRows[block]) {
                for (letter in 0 until PER_BLOCK) {
                    val cx = fit.columns[block * PER_BLOCK + letter].roundToInt()
                    val cy = fit.rows[row].roundToInt()
                    val mean = boxMean(data, stride, width, height, cx, cy, ring)
                    if (mean < 0.0) continue
                    values.add(mean)
                }
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

    /** Mean gray of the disc, or -1 when it does not fit on the page. */
    private fun discMean(
        pixels: ByteArray,
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
            val base = y * width
            for (segment in row.segments) {
                val x0 = centerX + segment[0]
                val x1 = centerX + segment[1]
                if (x0 < 0 || x1 >= width) return -1.0
                for (x in x0..x1) sum += (pixels[base + x].toInt() and 0xFF).toLong()
            }
        }
        return sum.toDouble() / kernel.area
    }

    /** Fraction of the disc whose pixels are darker than [threshold]. */
    private fun discInk(
        pixels: ByteArray,
        width: Int,
        height: Int,
        centerX: Int,
        centerY: Int,
        threshold: Double,
        kernel: Kernel,
    ): Double {
        var dark = 0
        for (row in kernel.rows) {
            val y = centerY + row.dy
            if (y < 0 || y >= height) return 0.0
            val base = y * width
            for (segment in row.segments) {
                val x0 = centerX + segment[0]
                val x1 = centerX + segment[1]
                if (x0 < 0 || x1 >= width) return 0.0
                for (x in x0..x1) if ((pixels[base + x].toInt() and 0xFF) < threshold) dark++
            }
        }
        return dark.toDouble() / kernel.area
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
