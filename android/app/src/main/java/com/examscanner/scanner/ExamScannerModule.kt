package com.examscanner.scanner

import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import java.io.File

/**
 * Native bridge for reading an answer sheet photo on-device.
 *
 * JS calls `scanPage(path)` with a photo captured by VisionCamera; the heavy OpenCV
 * work runs on a private background thread so the UI stays responsive.
 */
class ExamScannerModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val executor = java.util.concurrent.Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "exam-scanner").apply { priority = Thread.NORM_PRIORITY - 1 }
    }

    private var template: ExamTemplate? = null

    override fun getName(): String = NAME

    @ReactMethod
    fun scanPage(path: String, promise: Promise) {
        executor.execute {
            try {
                ensureOpenCV()
                val sheet = loadTemplate()
                val gray = AndroidImageLoader.loadGray(path)
                    ?: throw IllegalStateException("No se pudo decodificar la foto: $path")
                val debugFile = File(reactContext.cacheDir, "scan_debug.jpg")
                val outcome = try {
                    ScanEngine.scan(gray, sheet, debugFile, null)
                } finally {
                    gray.release()
                }
                promise.resolve(outcome.toMap())
            } catch (error: Throwable) {
                promise.reject("scan_failed", error.message ?: "No se pudo leer la hoja.", error)
            }
        }
    }

    private fun loadTemplate(): ExamTemplate {
        template?.let { return it }
        val json = reactContext.assets.open(TEMPLATE_ASSET).bufferedReader().use { it.readText() }
        return ExamTemplate.parse(json).also { template = it }
    }

    private fun ensureOpenCV() {
        if (openCvReady) return
        synchronized(ExamScannerModule::class.java) {
            if (openCvReady) return
            val loaded = try {
                org.opencv.android.OpenCVLoader.initLocal()
            } catch (_: Throwable) {
                false
            }
            if (!loaded) {
                try {
                    System.loadLibrary("opencv_java4")
                } catch (_: Throwable) {
                    // The error is reported when OpenCV is used, with a clearer stack trace.
                }
            }
            openCvReady = true
        }
    }

    private fun ScanOutcome.toMap() = Arguments.createMap().apply {
        putBoolean("ok", ok)
        putString("reason", reason)
        putString("message", message)
        putBoolean("pageFound", pageFound)
        putBoolean("gridFound", gridFound)
        putDouble("sharpness", sharpness)
        putDouble("brightness", brightness)
        putDouble("glare", glare)
        putDouble("coverage", coverage)
        putDouble("pageAreaRatio", pageAreaRatio)
        putInt("orientation", orientation)
        putDouble("elapsedMs", elapsedMs.toDouble())
        val answerArray = Arguments.createArray()
        for (answer in answers) {
            answerArray.pushMap(
                Arguments.createMap().apply {
                    putInt("number", answer.number)
                    putString("letter", answer.letter)
                    putDouble("ratio", answer.ratio)
                    putDouble("secondRatio", answer.secondRatio)
                    putBoolean("ambiguous", answer.ambiguous)
                },
            )
        }
        putArray("answers", answerArray)
        val warningArray = Arguments.createArray()
        for (warning in warnings) warningArray.pushString(warning)
        putArray("warnings", warningArray)
        debugPath?.let { putString("debugPath", it) }
    }

    override fun invalidate() {
        executor.shutdownNow()
        super.invalidate()
    }

    companion object {
        const val NAME = "ExamScanner"
        private const val TEMPLATE_ASSET = "template.json"

        @Volatile
        private var openCvReady = false
    }
}
