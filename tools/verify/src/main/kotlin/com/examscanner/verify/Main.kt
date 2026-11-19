package com.examscanner.verify

import com.examscanner.scanner.ExamTemplate
import com.examscanner.scanner.ScanEngine
import com.examscanner.scanner.ScanOutcome
import nu.pattern.OpenCV
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import java.util.Locale

private fun fmt(value: Double, pattern: String): String =
    String.format(Locale.US, pattern, value)

private fun jsonOutcome(outcome: ScanOutcome): String {
    val answers = outcome.answers.joinToString(",") { answer ->
        """{"number":${answer.number},"letter":"${answer.letter}","ratio":${fmt(answer.ratio, "%.3f")},"second":${fmt(answer.secondRatio, "%.3f")},"ambiguous":${answer.ambiguous}}"""
    }
    return """{"ok":${outcome.ok},"reason":"${outcome.reason}","pageFound":${outcome.pageFound},"gridFound":${outcome.gridFound},"sharpness":${fmt(outcome.sharpness, "%.1f")},"brightness":${fmt(outcome.brightness, "%.1f")},"glare":${fmt(outcome.glare, "%.3f")},"lattice":${fmt(outcome.latticeScore, "%.1f")},"coverage":${fmt(outcome.coverage, "%.3f")},"area":${fmt(outcome.pageAreaRatio, "%.3f")},"orientation":${outcome.orientation},"blocks":${outcome.blocks},"ms":${outcome.elapsedMs},"warnings":${outcome.warnings.size},"answers":[$answers]}"""
}

private fun answerLine(outcome: ScanOutcome): String =
    outcome.answers.joinToString(" ") { "${it.number}:${if (it.letter.isEmpty()) "." else it.letter}" }

/** A real capture with its hand-checked answers. */
private data class RealCase(
    val file: String,
    val blocks: Int,
    val answered: Map<Int, String>,
)

private fun scanFile(file: File, template: ExamTemplate, debug: Boolean, trace: Boolean): ScanOutcome {
    val image: Mat = Imgcodecs.imread(file.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)
    check(!image.empty()) { "Could not decode image: $file" }
    val debugFile = if (debug) File(file.parentFile, "${file.nameWithoutExtension}_debug.jpg") else null
    val logger: ((String) -> Unit)? = if (trace) ({ line -> println("trace $line") }) else null
    return try {
        ScanEngine.scan(image, template, debugFile, logger)
    } finally {
        image.release()
    }
}

fun main(args: Array<String>) {
    OpenCV.loadLocally()
    val root = File(args.getOrNull(2) ?: "../..").absoluteFile
    val templateFile = File(args.getOrNull(1) ?: "$root/android/app/src/main/assets/template.json")
    val template = ExamTemplate.parse(templateFile.readText())
    val verbose = System.getenv("VERIFY_TRACE") != null

    // Batch mode: scan every image in a directory and report, without fixture
    // expectations. Used to triage real captures pulled off a device.
    val input = File(args.getOrNull(0) ?: "$root/tools/verify/fixtures")
    if (input.isDirectory) {
        for (file in input.listFiles()!!.sortedBy { it.name }) {
            if (!file.name.endsWith(".jpg", true) && !file.name.endsWith(".png", true)) continue
            val image: Mat = Imgcodecs.imread(file.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)
            if (image.empty()) continue
            val outcome = ScanEngine.scan(
                image,
                template,
                null,
                if (verbose) ({ line -> println("  trace $line") }) else null,
            )
            image.release()
            val answers = outcome.answers.filter { it.letter.isNotEmpty() }
                .joinToString(",") { "${it.number}:${it.letter}${if (it.ambiguous) "?" else ""}" }
            println(
                "${file.name} ok=${outcome.ok} reason=${outcome.reason} page=${outcome.pageFound} " +
                    "rot=${outcome.orientation} blocks=${outcome.blocks} " +
                    "cov=${String.format(Locale.US, "%.2f", outcome.coverage)} " +
                    "area=${String.format(Locale.US, "%.2f", outcome.pageAreaRatio)} " +
                    "sharp=${String.format(Locale.US, "%.0f", outcome.sharpness)} " +
                    "bright=${String.format(Locale.US, "%.0f", outcome.brightness)} " +
                    "ms=${outcome.elapsedMs} ans=[$answers]",
            )
        }
        return
    }

    val fixturesDir = File("$root/tools/verify/fixtures")

    val failures = ArrayList<String>()

    // 1. Real captures. Ground truth checked by hand against the photos.
    val realCases = listOf(
        RealCase("test2.jpeg", blocks = 3, answered = mapOf(1 to "D")),
        RealCase("test3.jpeg", blocks = 2, answered = mapOf(1 to "A", 2 to "B", 3 to "B")),
    )
    for (case in realCases) {
        val photo = File(fixturesDir, case.file)
        if (!photo.exists()) {
            failures.add("${case.file}: missing")
            continue
        }
        val label = photo.nameWithoutExtension
        val photoImage: Mat = Imgcodecs.imread(photo.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)
        check(!photoImage.empty()) { "Could not decode image: $photo" }
        val warpMat = Mat()
        val outcome = try {
            ScanEngine.scan(
                photoImage,
                template,
                File(photo.parentFile, "${label}_debug.jpg"),
                if (verbose) ({ line -> println("trace $line") }) else null,
                warpMat,
            )
        } finally {
            photoImage.release()
        }
        if (!warpMat.empty()) {
            Imgcodecs.imwrite(File(photo.parentFile, "${label}_warp.png").absolutePath, warpMat)
        }
        warpMat.release()

        println("$label ${jsonOutcome(outcome)}")
        println("$label answers ${answerLine(outcome)}")
        if (!outcome.ok) failures.add("$label: expected ok, got ${outcome.reason}")
        for ((number, letter) in case.answered) {
            val got = outcome.answers.firstOrNull { it.number == number }?.letter
            if (got != letter) failures.add("$label: expected question $number = $letter, got $got")
        }
        val answered = outcome.answers.count { it.letter.isNotEmpty() }
        if (answered != case.answered.size) {
            failures.add("$label: expected ${case.answered.size} answered, got $answered (${answerLine(outcome)})")
        }
        if (outcome.blocks != case.blocks) failures.add("$label: expected ${case.blocks} blocks, got ${outcome.blocks}")
    }

    // 2. Synthetic fixtures: the rectified sheet pasted on a desk, rotated and degraded.
    val generatedDir = File(fixturesDir, "generated")
    val rectified = File(fixturesDir, "test2_warp.png")
    if (rectified.exists()) {
        val verbose = System.getenv("VERIFY_TRACE") != null
        for (fixture in Fixtures.build(rectified, template, generatedDir)) {
            if (verbose) println("fixture ${fixture.name} ----------")
            val outcome = scanFile(fixture.file, template, debug = verbose, trace = verbose)
            val question1 = outcome.answers.firstOrNull { it.number == 1 }?.letter
            val answered = outcome.answers.count { it.letter.isNotEmpty() }
            println(
                "fixture ${fixture.name} ok=${outcome.ok} reason=${outcome.reason} " +
                    "q1=$question1 answered=$answered blocks=${outcome.blocks} ms=${outcome.elapsedMs}",
            )
            if (outcome.ok != fixture.expectOk) {
                failures.add("fixture ${fixture.name}: expected ok=${fixture.expectOk}, got ok=${outcome.ok} (${outcome.reason})")
            }
            if (fixture.expectReason != null && !outcome.reason.startsWith(fixture.expectReason)) {
                failures.add("fixture ${fixture.name}: expected reason ${fixture.expectReason}, got ${outcome.reason}")
            }
            if (fixture.expectBlocks != null && outcome.blocks != fixture.expectBlocks) {
                failures.add("fixture ${fixture.name}: expected ${fixture.expectBlocks} blocks, got ${outcome.blocks}")
            }
            if (fixture.expectAnswers != null && outcome.answers.size != fixture.expectAnswers) {
                failures.add("fixture ${fixture.name}: expected ${fixture.expectAnswers} questions, got ${outcome.answers.size}")
            }
            if (fixture.expectOk) {
                if (question1 != "D") failures.add("fixture ${fixture.name}: expected question 1 = D, got $question1")
                if (answered != 1) failures.add("fixture ${fixture.name}: expected 1 answered, got $answered")
            }
        }
    } else {
        println("fixtures skipped: ${rectified.name} not found")
    }
    if (failures.isEmpty()) {
        println("PASS real=${realCases.size} fixtures")
    } else {
        failures.forEach { println("FAIL $it") }
        kotlin.system.exitProcess(1)
    }
}
