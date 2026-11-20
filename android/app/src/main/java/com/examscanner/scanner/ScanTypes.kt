package com.examscanner.scanner

import kotlin.math.abs
import kotlin.math.hypot

data class PointD(val x: Double, val y: Double) {
    fun distanceTo(other: PointD): Double = hypot(x - other.x, y - other.y)
}

data class QuadD(
    val tl: PointD,
    val tr: PointD,
    val br: PointD,
    val bl: PointD,
) {
    fun points(): List<PointD> = listOf(tl, tr, br, bl)

    fun rolled(steps: Int): QuadD {
        val points = points()
        val start = ((steps % 4) + 4) % 4
        return QuadD(
            points[start],
            points[(start + 1) % 4],
            points[(start + 2) % 4],
            points[(start + 3) % 4],
        )
    }

    fun center(): PointD {
        val points = points()
        return PointD(points.sumOf { it.x } / 4.0, points.sumOf { it.y } / 4.0)
    }

    fun area(): Double {
        val points = points()
        var sum = 0.0
        for (index in points.indices) {
            val current = points[index]
            val next = points[(index + 1) % 4]
            sum += current.x * next.y - next.x * current.y
        }
        return abs(sum) / 2.0
    }

    fun minSide(): Double = points().indices.minOf { index ->
        points()[index].distanceTo(points()[(index + 1) % 4])
    }
}

data class GridBox(
    val x: Double,
    val y: Double,
    val width: Double,
    val height: Double,
)

data class OutcomeAnswer(
    val number: Int,
    val letter: String,
    val ratio: Double,
    val secondRatio: Double,
    val ambiguous: Boolean,
)

data class ScanOutcome(
    val ok: Boolean,
    val reason: String,
    val message: String,
    val answers: List<OutcomeAnswer>,
    val warnings: List<String>,
    val pageFound: Boolean,
    val gridFound: Boolean,
    val sharpness: Double,
    val brightness: Double,
    val glare: Double,
    val latticeScore: Double,
    val coverage: Double,
    val pageAreaRatio: Double,
    val orientation: Int,
    /** Printed blocks (A-E groups) found on the sheet: 1 to 6. */
    val blocks: Int,
    val elapsedMs: Long,
    val debugPath: String? = null,
    val error: String? = null,
    /** Content of the QR code printed in the sheet header, when one was found. */
    val qr: String? = null,
)
