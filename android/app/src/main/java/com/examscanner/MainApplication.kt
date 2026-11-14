package com.examscanner

import android.app.Application
import com.examscanner.scanner.ExamScannerPackage
import com.examscanner.scanner.ScanPagePlugin
import com.facebook.react.PackageList
import com.facebook.react.ReactApplication
import com.facebook.react.ReactHost
import com.facebook.react.ReactNativeApplicationEntryPoint.loadReactNative
import com.facebook.react.ReactNativeHost
import com.facebook.react.ReactPackage
import com.facebook.react.defaults.DefaultReactHost.getDefaultReactHost
import com.facebook.react.defaults.DefaultReactNativeHost
import com.mrousavy.camera.frameprocessors.FrameProcessorPluginRegistry

class MainApplication : Application(), ReactApplication {

  override val reactNativeHost: ReactNativeHost =
      object : DefaultReactNativeHost(this) {
        override fun getPackages(): List<ReactPackage> =
            PackageList(this).packages.apply {
              add(ExamScannerPackage())
            }

        override fun getJSMainModuleName(): String = "index"

        override fun getUseDeveloperSupport(): Boolean = BuildConfig.DEBUG

        override val isNewArchEnabled: Boolean = BuildConfig.IS_NEW_ARCHITECTURE_ENABLED
        override val isHermesEnabled: Boolean = BuildConfig.IS_HERMES_ENABLED
      }

  override val reactHost: ReactHost
    get() = getDefaultReactHost(applicationContext, reactNativeHost)

  companion object {
    init {
      try {
        System.loadLibrary("opencv_java4")
      } catch (e: Throwable) {
        android.util.Log.e("MainApplication", "Failed to load opencv_java4", e)
      }
    }
  }

  override fun onCreate() {
    super.onCreate()
    try {
      System.loadLibrary("opencv_java4")
    } catch (e: Throwable) {
      android.util.Log.e("MainApplication", "Failed to load opencv_java4 in onCreate", e)
    }
    FrameProcessorPluginRegistry.addFrameProcessorPlugin(ScanPagePlugin.NAME) { proxy, options ->
      ScanPagePlugin(proxy, options)
    }
    loadReactNative(this)
  }
}
