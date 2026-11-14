plugins {
    kotlin("jvm") version "2.1.20"
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.openpnp:opencv:4.9.0-0")
    implementation("org.json:json:20240303")
}

kotlin {
    jvmToolchain(17)
}

sourceSets {
    main {
        kotlin.srcDir("../../android/app/src/main/java/com/examscanner/scanner")
        kotlin.exclude(
            "AndroidImageLoader.kt",
            "ExamScannerModule.kt",
            "ExamScannerPackage.kt",
            "ScanPagePlugin.kt",
        )
        kotlin.srcDir("src/main/kotlin")
    }
}

application {
    mainClass.set("com.examscanner.verify.MainKt")
}
