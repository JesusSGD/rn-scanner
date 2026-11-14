package com.examscanner.scanner

import org.json.JSONObject
import kotlin.math.max
import kotlin.math.min

data class Bubble(
    val letter: Char,
    val x: Double,
    val y: Double,
)

data class Question(
    val number: Int,
    val options: List<Bubble>,
)

data class ExamTemplate(
    val pageWidth: Int,
    val pageHeight: Int,
    val gridRoi: GridBox,
    val bubbleRadius: Double,
    val questions: List<Question>,
) {
    companion object {
        private const val LETTERS = "ABCDE"

        fun parse(json: String): ExamTemplate {
            val root = JSONObject(json)
            val questionsJson = root.getJSONArray("questions")
            val questions = ArrayList<Question>(questionsJson.length())
            for (index in 0 until questionsJson.length()) {
                val questionJson = questionsJson.getJSONObject(index)
                val optionsJson = questionJson.getJSONObject("options")
                val options = ArrayList<Bubble>(LETTERS.length)
                for (letter in LETTERS) {
                    val point = optionsJson.getJSONArray(letter.toString())
                    options.add(Bubble(letter, point.getDouble(0), point.getDouble(1)))
                }
                questions.add(Question(questionJson.getInt("number"), options))
            }
            questions.sortBy { it.number }

            val bubbleRadius = root.getDouble("bubble_radius")
            val gridRoi = if (root.has("grid_roi")) {
                val roi = root.getJSONArray("grid_roi")
                GridBox(
                    roi.getDouble(0),
                    roi.getDouble(1),
                    roi.getDouble(2),
                    roi.getDouble(3),
                )
            } else {
                bubbleSpan(questions, bubbleRadius * 2.0)
            }

            return ExamTemplate(
                pageWidth = root.getInt("page_width"),
                pageHeight = root.getInt("page_height"),
                gridRoi = gridRoi,
                bubbleRadius = bubbleRadius,
                questions = questions,
            )
        }

        private fun bubbleSpan(questions: List<Question>, pad: Double): GridBox {
            var left = Double.MAX_VALUE
            var top = Double.MAX_VALUE
            var right = -Double.MAX_VALUE
            var bottom = -Double.MAX_VALUE
            for (question in questions) {
                for (option in question.options) {
                    left = min(left, option.x)
                    top = min(top, option.y)
                    right = max(right, option.x)
                    bottom = max(bottom, option.y)
                }
            }
            return GridBox(left - pad, top - pad, right - left + 2 * pad, bottom - top + 2 * pad)
        }
    }
}
