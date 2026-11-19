package com.examscanner.scanner

import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.Locale
import kotlin.math.max
import kotlin.math.roundToInt

object ScanEngine {
    private const val MIN_PAGE_AREA = 0.12
    private const val MIN_SHARPNESS = 8.0
    private const val MIN_BRIGHTNESS = 65.0
    // Paper-white scans reach a mean of 255 with crisp print; a truly blown-out photo
    // loses the lattice and fails the coverage gate, which is the better message.
    private const val MAX_BRIGHTNESS = 255.0
    private const val MAX_GLARE = 0.90
    // 125/165 cells is a solid fit on a blurry desk photo; the remaining cells'
    // positions come from the fitted lattice itself. The gate is a fraction because
    // the sheet may print one to six blocks with a partial last block.
    private const val MIN_COVERAGE = 0.75
    private const val MIN_RING_STRENGTH = 12.0
    private const val ORIENTATION_SCALE = 0.75

    fun scan(
        gray: Mat,
        template: ExamTemplate,
        debugFile: File? = null,
        trace: ((String) -> Unit)? = null,
        warpOut: Mat? = null,
    ): ScanOutcome {
        val started = System.nanoTime()
        val elapsed = { (System.nanoTime() - started) / 1_000_000 }

        if (gray.empty() || gray.cols() < 64 || gray.rows() < 64) {
            return failure("error", "No se pudo leer la imagen.", elapsed())
        }

        val page = PageDetector.detect(gray)
            ?: return failure(
                "no_page",
                "No se detecta la hoja. Apunta la cámara al examen y asegúrate de ver los cuatro bordes.",
                elapsed(),
                pageFound = false,
            )
        trace?.invoke("t page=${elapsed()}")
        val pageAreaRatio = page.area() / (gray.cols().toDouble() * gray.rows().toDouble())
        if (pageAreaRatio < MIN_PAGE_AREA) {
            return failure(
                "page_too_small",
                "Acerca la hoja: debe ocupar la mayor parte del encuadre.",
                elapsed(),
                pageFound = true,
                pageAreaRatio = pageAreaRatio,
            )
        }
        trace?.invoke(String.format(Locale.US, "page area=%.3f", pageAreaRatio))

        // Pick the page orientation before paying for a full-resolution fit. Evaluation
        // halves keep the loop cheap, and the asymmetry of the printed grid (the last
        // block may print only a few rows) makes the right orientation win by a wide
        // margin. A decisive fit ends the search early: full coverage, no foreign rings
        // and the grid where the sheet template puts it.
        var bestRotation = -1
        var bestScore = 0.0
        for (rotation in 0 until 4) {
            val warp = warpToPage(gray, page.rolled(rotation), template, ORIENTATION_SCALE)
            val fit = AnswerGrid.fit(warp, template, trace)
            warp.release()
            val score = fit?.latticeScore ?: 0.0
            trace?.invoke("t rot$rotation=${elapsed()}")
            if (trace != null && fit != null) {
                trace(
                    String.format(
                        Locale.US,
                        "rot=%d cells=%d/%d extra=%d cols=%d rows=%d blocks=%d radius=%.1f score=%.1f",
                        rotation,
                        fit.matchedCells,
                        fit.totalCells,
                        fit.extraCells,
                        fit.matchedColumns,
                        fit.matchedRows,
                        fit.blockRows.size,
                        fit.radius,
                        score,
                    ),
                )
            }
            if (fit != null && score > bestScore) {
                bestRotation = rotation
                bestScore = score
            }
            // A decisive fit ends the search early; foreign rings mean the lattice is
            // explaining only part of the page, and a grid far from the template's page
            // position means the sheet is probably being read upside down.
            if (fit != null && fit.coverage >= 0.9 && fit.extraCells == 0 && fit.positionError < 0.05) break
        }
        val rotation = if (bestRotation >= 0) bestRotation else 0

        // The small-scale warp reads blurry desk photos far better (downscaling shrinks
        // the relative blur), so the first fit happens there. When it is not decisive,
        // the full-scale warp is tried too and the better lattice wins; its coordinates
        // are mapped onto the full-scale warp, which keeps answer reading at full detail.
        val smallWarp = warpToPage(gray, page.rolled(rotation), template, ORIENTATION_SCALE)
        val smallFit = AnswerGrid.fit(smallWarp, template, trace)
        smallWarp.release()
        var fit = smallFit
        var fitScale = 1.0 / ORIENTATION_SCALE
        if (smallFit == null || smallFit.coverage < 0.85) {
            val fullWarp = warpToPage(gray, page.rolled(rotation), template, 1.0)
            val fullFit = AnswerGrid.fit(fullWarp, template, trace)
            fullWarp.release()
            if (fullFit != null && (smallFit == null || fullFit.latticeScore > smallFit.latticeScore)) {
                fit = fullFit
                fitScale = 1.0
            }
        }
        fit = fit?.scaled(fitScale)
        val warp = warpToPage(gray, page.rolled(rotation), template, 1.0)
        val ringStrength = if (fit != null) AnswerGrid.ringStrength(warp, fit) else 0.0
        val sharpness = PageDetector.sharpness(warp)
        val brightness = PageDetector.brightness(warp)
        val glare = PageDetector.glareRatio(warp)
        trace?.invoke("t quality=${elapsed()}")
        warpOut?.let { warp.copyTo(it) }
        trace?.invoke(
            String.format(
                Locale.US,
                "chosen rot=%d coverage=%.2f ring=%.1f sharpness=%.1f brightness=%.1f glare=%.3f",
                rotation,
                fit?.coverage ?: 0.0,
                ringStrength,
                sharpness,
                brightness,
                glare,
            ),
        )

        val reasons = ArrayList<String>(4)
        val messages = ArrayList<String>(4)
        if (brightness < MIN_BRIGHTNESS) {
            reasons.add("too_dark")
            messages.add("Hay poca luz. Enciende la linterna o busca un lugar más claro.")
        }
        if (brightness > MAX_BRIGHTNESS) {
            reasons.add("too_bright")
            messages.add("La hoja está sobreexpuesta. Evita la luz directa.")
        }
        if (glare > MAX_GLARE) {
            reasons.add("glare")
            messages.add("Hay reflejos sobre la hoja. Cambia un poco el ángulo.")
        }
        if (sharpness < MIN_SHARPNESS) {
            reasons.add("blurry")
            messages.add("La foto está borrosa. Mantén el pulso y espera a que la cámara enfoque.")
        }
        if (fit == null) {
            reasons.add("no_pattern")
            messages.add("No se reconoce la hoja de respuestas. Verifica que sea la hoja correcta y que se vea completa.")
        } else {
            if (fit.coverage < MIN_COVERAGE) {
                reasons.add("no_pattern")
                messages.add("No se reconoce el patrón de burbujas. Verifica que sea la hoja correcta y que se vea completa.")
            }
            if (ringStrength < MIN_RING_STRENGTH) {
                reasons.add("no_pattern")
                messages.add("Las burbujas no se ven con nitidez. Acerca la hoja y mejora la iluminación.")
            }
        }

        if (fit == null) {
            warp.release()
            return ScanOutcome(
                ok = false,
                reason = reasons.first(),
                message = messages.first(),
                answers = emptyList(),
                warnings = emptyList(),
                pageFound = true,
                gridFound = false,
                sharpness = sharpness,
                brightness = brightness,
                glare = glare,
                latticeScore = ringStrength,
                coverage = 0.0,
                pageAreaRatio = pageAreaRatio,
                orientation = rotation,
                blocks = 0,
                elapsedMs = elapsed(),
            )
        }

        val answers = AnswerGrid.read(warp, fit)

        val warnings = ArrayList<String>()
        if (fit.matchedColumns < fit.columns.size || fit.matchedRows < fit.rows.size) {
            warnings.add("Algunas burbujas no se detectaron; revisa las marcas dudosas.")
        }
        val ambiguous = answers.filter { it.ambiguous }
        ambiguous.take(5).forEach { answer ->
            warnings.add("Pregunta ${answer.number}: marca dudosa o doble.")
        }
        if (ambiguous.size > 5) {
            warnings.add("Y ${ambiguous.size - 5} preguntas más con marcas dudosas.")
        }
        val answered = answers.count { it.letter.isNotEmpty() }
        if (answered == 0) {
            warnings.add("No se detectó ninguna respuesta marcada.")
        }

        val ok = reasons.isEmpty()
        val debugPath = if (debugFile != null) {
            writeDebug(warp, fit, answers, debugFile)
        } else {
            null
        }
        val message = if (ok) "Hoja leída correctamente." else messages.first()
        warp.release()

        return ScanOutcome(
            ok = ok,
            reason = if (ok) "ok" else reasons.first(),
            message = message,
            answers = answers,
            warnings = warnings,
            pageFound = true,
            gridFound = fit.coverage >= MIN_COVERAGE,
            sharpness = sharpness,
            brightness = brightness,
            glare = glare,
            latticeScore = ringStrength,
            coverage = fit.coverage,
            pageAreaRatio = pageAreaRatio,
            orientation = rotation,
            blocks = fit.blockRows.size,
            elapsedMs = elapsed(),
            debugPath = debugPath,
        )
    }

    private fun warpToPage(
        gray: Mat,
        quad: QuadD,
        template: ExamTemplate,
        scale: Double = 1.0,
    ): Mat {
        val width = (template.pageWidth * scale).roundToInt()
        val height = (template.pageHeight * scale).roundToInt()
        val source = MatOfPoint2f(*quad.points().map { Point(it.x, it.y) }.toTypedArray())
        val destination = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(width - 1.0, 0.0),
            Point(width - 1.0, height - 1.0),
            Point(0.0, height - 1.0),
        )
        val matrix = Imgproc.getPerspectiveTransform(source, destination)
        val warped = Mat()
        Imgproc.warpPerspective(
            gray,
            warped,
            matrix,
            Size(width.toDouble(), height.toDouble()),
            Imgproc.INTER_LINEAR,
        )
        matrix.release()
        source.release()
        destination.release()
        return warped
    }

    private fun writeDebug(
        warp: Mat,
        fit: AnswerGrid.Fit,
        answers: List<OutcomeAnswer>,
        file: File,
    ): String? = try {
        file.parentFile?.mkdirs()
        val debug = Mat()
        Imgproc.cvtColor(warp, debug, Imgproc.COLOR_GRAY2BGR)
        val answerByNumber = answers.associateBy { it.number }
        val patternStroke = max(2, (fit.radius * 0.3).roundToInt())
        for (block in fit.blockRows.indices) {
            for (row in 0 until fit.blockRows[block]) {
                for (letter in 0 until AnswerGrid.PER_BLOCK) {
                    Imgproc.circle(
                        debug,
                        Point(
                            fit.columns[block * AnswerGrid.PER_BLOCK + letter],
                            fit.rows[row],
                        ),
                        max(2, (fit.radius * 0.8).roundToInt()),
                        Scalar(255.0, 190.0, 115.0),
                        patternStroke,
                    )
                }
            }
        }

        // Answer-grid box: the fitted lattice bounds, drawn so the photo review shows
        // exactly what was treated as the answer area.
        val pad = fit.radius * 1.6
        val left = (fit.columns.min() - pad).coerceAtLeast(0.0)
        val top = (fit.rows.min() - pad).coerceAtLeast(0.0)
        val right = (fit.columns.max() + pad).coerceAtMost(warp.cols() - 1.0)
        val bottom = (fit.rows.max() + pad).coerceAtMost(warp.rows() - 1.0)
        Imgproc.rectangle(
            debug,
            Point(left, top),
            Point(right, bottom),
            Scalar(255.0, 166.0, 88.0),
            max(3, (fit.radius * 0.5).roundToInt()),
        )
        Imgproc.putText(
            debug,
            "Cuadro de respuestas",
            Point(left + pad * 0.2, max(14.0, top - 8.0)),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            max(0.6, fit.radius / 14.0),
            Scalar(255.0, 166.0, 88.0),
            3,
        )

        for (block in fit.blockRows.indices) {
            for (row in 0 until fit.blockRows[block]) {
                val answer = answerByNumber[block * AnswerGrid.QUESTIONS_PER_BLOCK + row + 1]
                for (letter in 0 until AnswerGrid.PER_BLOCK) {
                    val x = fit.columns[block * AnswerGrid.PER_BLOCK + letter]
                    val y = fit.rows[row]
                    val color = when {
                        answer == null -> Scalar(0.0, 0.0, 255.0)
                        AnswerGrid.LETTERS[letter].toString() == answer.letter -> Scalar(0.0, 220.0, 0.0)
                        else -> Scalar(170.0, 170.0, 170.0)
                    }
                    Imgproc.circle(
                        debug,
                        Point(x, y),
                        max(3, (fit.radius * 0.8).roundToInt()),
                        color,
                        1,
                    )
                }
                if (answer != null && answer.letter.isNotEmpty()) {
                    val index = AnswerGrid.LETTERS.indexOf(answer.letter)
                    if (index >= 0) {
                        Imgproc.putText(
                            debug,
                            answer.letter,
                            Point(
                                fit.columns[block * AnswerGrid.PER_BLOCK + index] + fit.radius + 2,
                                fit.rows[row] + 5,
                            ),
                            Imgproc.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            Scalar(0.0, 200.0, 0.0),
                            1,
                        )
                    }
                }
            }
        }
        Imgcodecs.imwrite(file.absolutePath, debug)
        debug.release()
        file.absolutePath
    } catch (_: Throwable) {
        null
    }

    private fun failure(
        reason: String,
        message: String,
        elapsedMs: Long,
        pageFound: Boolean = false,
        pageAreaRatio: Double = 0.0,
    ) = ScanOutcome(
        ok = false,
        reason = reason,
        message = message,
        answers = emptyList(),
        warnings = emptyList(),
        pageFound = pageFound,
        gridFound = false,
        sharpness = 0.0,
        brightness = 0.0,
        glare = 0.0,
        latticeScore = 0.0,
        coverage = 0.0,
        pageAreaRatio = pageAreaRatio,
        orientation = 0,
        blocks = 0,
        elapsedMs = elapsedMs,
    )
}
