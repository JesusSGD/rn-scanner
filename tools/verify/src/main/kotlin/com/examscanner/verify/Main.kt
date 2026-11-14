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
    return """{"ok":${outcome.ok},"reason":"${outcome.reason}","pageFound":${outcome.pageFound},"gridFound":${outcome.gridFound},"sharpness":${fmt(outcome.sharpness, "%.1f")},"brightness":${fmt(outcome.brightness, "%.1f")},"glare":${fmt(outcome.glare, "%.3f")},"lattice":${fmt(outcome.latticeScore, "%.1f")},"coverage":${fmt(outcome.coverage, "%.3f")},"area":${fmt(outcome.pageAreaRatio, "%.3f")},"orientation":${outcome.orientation},"ms":${outcome.elapsedMs},"warnings":${outcome.warnings.size},"answers":[$answers]}"""
}

private fun answerLine(outcome: ScanOutcome): String =
    outcome.answers.joinToString(" ") { "${it.number}:${if (it.letter.isEmpty()) "." else it.letter}" }

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
    val fixturesDir = File("$root/tools/verify/fixtures")
    val template = ExamTemplate.parse(templateFile.readText())
    val verbose = System.getenv("VERIFY_TRACE") != null

    val failures = ArrayList<String>()

    // 1. The real photo. Ground truth: only question 1 is answered, with D.
    val photo = File(args.getOrNull(0) ?: "$fixturesDir/test2.jpeg")
    val photoImage: Mat = Imgcodecs.imread(photo.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)
    check(!photoImage.empty()) { "Could not decode image: $photo" }
    val rectified = File(fixturesDir, "test2_warp.png")
    val rectifiedMat = Mat()
    val photoOutcome = try {
        ScanEngine.scan(
            photoImage,
            template,
            File(photo.parentFile, "${photo.nameWithoutExtension}_debug.jpg"),
            if (verbose) ({ line -> println("trace $line") }) else null,
            rectifiedMat,
        )
    } finally {
        photoImage.release()
    }
    if (!rectifiedMat.empty()) Imgcodecs.imwrite(rectified.absolutePath, rectifiedMat)
    rectifiedMat.release()

    println("test2 ${jsonOutcome(photoOutcome)}")
    println("test2 answers ${answerLine(photoOutcome)}")
    run {
        val question1 = photoOutcome.answers.firstOrNull { it.number == 1 }?.letter
        val answered = photoOutcome.answers.count { it.letter.isNotEmpty() }
        if (!photoOutcome.ok) failures.add("test2: expected ok, got ${photoOutcome.reason}")
        if (question1 != "D") failures.add("test2: expected question 1 = D, got $question1")
        if (answered != 1) failures.add("test2: expected 1 answered, got $answered")
    }

    // 2. Synthetic fixtures: the rectified sheet pasted on a desk, rotated and degraded.
    val generatedDir = File(fixturesDir, "generated")
    if (rectified.exists()) {
        val verbose = System.getenv("VERIFY_TRACE") != null
        for (fixture in Fixtures.build(rectified, generatedDir)) {
            if (verbose) println("fixture ${fixture.name} ----------")
            val outcome = scanFile(fixture.file, template, debug = verbose, trace = verbose)
            val question1 = outcome.answers.firstOrNull { it.number == 1 }?.letter
            val answered = outcome.answers.count { it.letter.isNotEmpty() }
            println(
                "fixture ${fixture.name} ok=${outcome.ok} reason=${outcome.reason} " +
                    "q1=$question1 answered=$answered ms=${outcome.elapsedMs}",
            )
            if (outcome.ok != fixture.expectOk) {
                failures.add("fixture ${fixture.name}: expected ok=${fixture.expectOk}, got ok=${outcome.ok} (${outcome.reason})")
            }
            if (fixture.expectReason != null && !outcome.reason.startsWith(fixture.expectReason)) {
                failures.add("fixture ${fixture.name}: expected reason ${fixture.expectReason}, got ${outcome.reason}")
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
        println("PASS ${args.getOrNull(0) ?: "test2.jpeg"}")
    } else {
        failures.forEach { println("FAIL $it") }
        kotlin.system.exitProcess(1)
    }
}
