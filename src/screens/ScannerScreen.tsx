import type {ScanResult} from '../native/ExamScanner';
import {scanPage} from '../native/ExamScanner';
import React, {useCallback, useMemo, useRef, useState} from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  useWindowDimensions,
} from 'react-native';
import Svg, {Polygon} from 'react-native-svg';
import {useSafeAreaInsets} from 'react-native-safe-area-context';
import {
  Camera,
  runAtTargetFps,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  VisionCameraProxy,
} from 'react-native-vision-camera';
import {useRunOnJS, useSharedValue} from 'react-native-worklets-core';

type Props = {
  onResult: (result: ScanResult) => void;
};

type LiveState = {
  status: number;
  ready: boolean;
  stable: number;
  // Quad corners in screen-relative 0..1 coords (tl,tr,br,bl x,y)
  quad: number[] | null;
  // Frame dimensions when quad was captured (to map to screen)
  frameW: number;
  frameH: number;
};

const REQUIRED_STABLE = 4;

const STATUS_MESSAGES: Record<number, string> = {
  0: 'Buscando la hoja…',
  1: 'Hay poca luz. Enciende la linterna.',
  2: 'Demasiada luz. Evita el reflejo directo.',
  3: 'Reflejo sobre la hoja. Cambia un poco el ángulo.',
  4: 'Enfocando… mantén el pulso.',
  5: 'Acerca la hoja para que ocupe el recuadro.',
  6: 'Muestra los cuatro bordes de la hoja.',
  7: 'No muevas…',
};

const wait = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

export default function ScannerScreen({onResult}: Props) {
  const insets = useSafeAreaInsets();
  const {width: screenW, height: screenH} = useWindowDimensions();
  const {hasPermission, requestPermission} = useCameraPermission();
  const device = useCameraDevice('back');
  const cameraRef = useRef<Camera>(null);

  const [live, setLive] = useState<LiveState>({
    status: 0,
    ready: false,
    stable: 0,
    quad: null,
    frameW: 1,
    frameH: 1,
  });
  const [torch, setTorch] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [hint, setHint] = useState('');
  const failures = useRef(0);

  const busy = useSharedValue(0);
  const captureRef = useRef<() => void>(() => {});

  const plugin = useMemo(() => {
    try {
      return VisionCameraProxy.initFrameProcessorPlugin('scanPage', {});
    } catch {
      return undefined;
    }
  }, []);

  const onLive = useRunOnJS(
    (
      status: number,
      ready: number,
      stable: number,
      quadPoints: number[],
      frameW: number,
      frameH: number,
    ) => {
      setLive(previous => {
        const nextQuad = quadPoints.length === 8 ? quadPoints : null;
        if (
          previous.status === status &&
          previous.ready === (ready === 1) &&
          previous.stable === stable &&
          previous.quad === nextQuad
        ) {
          return previous;
        }
        return {status, ready: ready === 1, stable, quad: nextQuad, frameW, frameH};
      });
    },
    [],
  );

  const onReady = useRunOnJS(() => {
    captureRef.current();
  }, []);

  const frameProcessor = useFrameProcessor(
    frame => {
      'worklet';
      if (plugin == null || busy.value === 1) {
        return;
      }
      const output = runAtTargetFps(8, () => {
        'worklet';
        return plugin.call(frame);
      });
      if (output == null) {
        return;
      }
      const values = output as number[];
      if (values.length < 3) {
        return;
      }
      // values: [status, ready, stable, areaRatio, brightness, glare, sharpness, tlX, tlY, trX, trY, brX, brY, blX, blY]
      const quad: number[] = values.length >= 15
        ? [values[7], values[8], values[9], values[10], values[11], values[12], values[13], values[14]]
        : [];
      onLive(values[0], values[1], values[2], quad, frame.width, frame.height);
      if (values[1] === 1) {
        onReady();
      }
    },
    [plugin, busy, onLive, onReady],
  );

  const capture = useCallback(async () => {
    if (busy.value === 1) {
      return;
    }
    const camera = cameraRef.current;
    if (!camera) {
      return;
    }
    busy.value = 1;
    setProcessing(true);
    setHint('');
    try {
      const photo = await camera.takePhoto({
        flash: 'off',
        enableShutterSound: false,
      });
      const result = await scanPage(photo.path);
      if (result.ok) {
        onResult(result);
        return;
      }
      failures.current += 1;
      if (failures.current >= 2) {
        onResult(result);
        return;
      }
      setHint(result.message);
      setProcessing(false);
      await wait(1000);
    } catch (error) {
      setHint(
        error instanceof Error
          ? error.message
          : 'No se pudo leer la hoja. Inténtalo otra vez.',
      );
      setProcessing(false);
      await wait(600);
    } finally {
      busy.value = 0;
    }
  }, [busy, onResult]);

  captureRef.current = capture;

  if (!hasPermission) {
    return (
      <View style={styles.centered}>
        <Text style={styles.permissionTitle}>Se necesita la cámara</Text>
        <Text style={styles.permissionText}>
          El examen se lee en el teléfono: ninguna foto sale del dispositivo.
        </Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Permitir cámara</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (device == null) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator color="#3fb950" size="large" />
        <Text style={styles.permissionText}>Iniciando la cámara…</Text>
      </View>
    );
  }

  const isReady = live.status === 7;
  const statusText = isReady
    ? live.ready
      ? 'Capturando…'
      : STATUS_MESSAGES[7]
    : STATUS_MESSAGES[live.status] ?? STATUS_MESSAGES[0];

  // Build SVG polygon points from frame-space quad → screen coords
  const quadPolygon = useMemo(() => {
    const q = live.quad;
    if (!q || q.length < 8 || live.frameW <= 0 || live.frameH <= 0) {
      return null;
    }
    // The camera frame is in landscape internally but rendered portrait on screen.
    // VisionCamera handles the rotation, so the quad coords (from the downsampled
    // luma plane) are already in display orientation: x→width, y→height of preview.
    const scaleX = screenW / live.frameW;
    const scaleY = screenH / live.frameH;
    const pts = [
      {x: q[0] * scaleX, y: q[1] * scaleY},   // TL
      {x: q[2] * scaleX, y: q[3] * scaleY},   // TR
      {x: q[4] * scaleX, y: q[5] * scaleY},   // BR
      {x: q[6] * scaleX, y: q[7] * scaleY},   // BL
    ];
    return pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
  }, [live.quad, live.frameW, live.frameH, screenW, screenH]);

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        photo
        pixelFormat="yuv"
        torch={torch ? 'on' : 'off'}
        frameProcessor={frameProcessor}
        enableZoomGesture
      />

      <View style={[styles.topBar, {paddingTop: insets.top + 12}]}>
        <View style={styles.topTextBlock}>
          <Text style={styles.title}>Escáner de respuestas</Text>
          <Text style={styles.subtitle}>La foto se toma sola cuando la hoja se ve bien</Text>
        </View>
        {device.hasTorch ? (
          <TouchableOpacity
            style={[styles.torch, torch && styles.torchOn]}
            onPress={() => setTorch(value => !value)}>
            <Text style={styles.torchText}>{torch ? 'Luz' : 'Linterna'}</Text>
          </TouchableOpacity>
        ) : null}
      </View>

      <View pointerEvents="none" style={styles.guideLayer}>
        <View style={[styles.guide, isReady && styles.guideReady]}>
          <View style={[styles.corner, styles.cornerTopLeft]} />
          <View style={[styles.corner, styles.cornerTopRight]} />
          <View style={[styles.corner, styles.cornerBottomLeft]} />
          <View style={[styles.corner, styles.cornerBottomRight]} />
        </View>
      </View>

      {/* Live quad overlay — only shown when a page is detected */}
      {quadPolygon ? (
        <Svg
          pointerEvents="none"
          style={StyleSheet.absoluteFill}
          width={screenW}
          height={screenH}>
          <Polygon
            points={quadPolygon}
            fill={isReady ? 'rgba(63,185,80,0.12)' : 'rgba(255,200,0,0.10)'}
            stroke={isReady ? '#3fb950' : '#d29922'}
            strokeWidth={2.5}
          />
        </Svg>
      ) : null}

      <View style={[styles.bottom, {paddingBottom: insets.bottom + 20}]}>
        {hint.length > 0 ? (
          <View style={styles.hint}>
            <Text style={styles.hintText}>{hint}</Text>
          </View>
        ) : null}

        <View style={[styles.status, isReady && styles.statusReady]}>
          <View style={styles.statusRow}>
            {!isReady ? (
              <ActivityIndicator color="#d29922" size="small" />
            ) : null}
            <Text style={[styles.statusText, isReady && styles.statusTextReady]}>
              {statusText}
            </Text>
          </View>
          {isReady ? (
            <View style={styles.dots}>
              {Array.from({length: REQUIRED_STABLE}).map((_, index) => (
                <View
                  key={index}
                  style={[
                    styles.dot,
                    index < Math.min(live.stable, REQUIRED_STABLE) && styles.dotOn,
                  ]}
                />
              ))}
            </View>
          ) : null}
        </View>

        <Text style={styles.footer}>
          Todo se procesa en el teléfono. Funciona sin internet.
        </Text>
      </View>

      {processing ? (
        <View style={styles.processing}>
          <ActivityIndicator color="#3fb950" size="large" />
          <Text style={styles.processingText}>Leyendo respuestas…</Text>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  centered: {
    flex: 1,
    backgroundColor: '#0d1117',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 32,
  },
  permissionTitle: {
    color: '#f0f6fc',
    fontSize: 22,
    fontWeight: '700',
    marginBottom: 10,
    textAlign: 'center',
  },
  permissionText: {
    color: '#9aa4af',
    fontSize: 15,
    lineHeight: 22,
    textAlign: 'center',
    marginBottom: 12,
  },
  button: {
    marginTop: 12,
    backgroundColor: '#238636',
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 28,
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '700',
  },
  topBar: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    paddingHorizontal: 20,
    paddingBottom: 12,
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(0,0,0,0.42)',
  },
  topTextBlock: {
    flex: 1,
    paddingRight: 12,
  },
  title: {
    color: '#ffffff',
    fontSize: 20,
    fontWeight: '700',
  },
  subtitle: {
    color: '#c9d1d9',
    fontSize: 12,
    marginTop: 4,
  },
  torch: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.45)',
    backgroundColor: 'rgba(0,0,0,0.35)',
  },
  torchOn: {
    backgroundColor: '#d29922',
    borderColor: '#d29922',
  },
  torchText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '600',
  },
  guideLayer: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  guide: {
    width: '76%',
    maxWidth: 420,
    aspectRatio: 0.7,
    borderWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.55)',
    borderRadius: 18,
  },
  guideReady: {
    borderColor: 'rgba(63,185,80,0.9)',
    backgroundColor: 'rgba(63,185,80,0.05)',
  },
  corner: {
    position: 'absolute',
    width: 34,
    height: 34,
    borderColor: '#3fb950',
  },
  cornerTopLeft: {
    top: -2,
    left: -2,
    borderTopWidth: 4,
    borderLeftWidth: 4,
    borderTopLeftRadius: 18,
  },
  cornerTopRight: {
    top: -2,
    right: -2,
    borderTopWidth: 4,
    borderRightWidth: 4,
    borderTopRightRadius: 18,
  },
  cornerBottomLeft: {
    bottom: -2,
    left: -2,
    borderBottomWidth: 4,
    borderLeftWidth: 4,
    borderBottomLeftRadius: 18,
  },
  cornerBottomRight: {
    bottom: -2,
    right: -2,
    borderBottomWidth: 4,
    borderRightWidth: 4,
    borderBottomRightRadius: 18,
  },
  bottom: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    paddingHorizontal: 20,
    alignItems: 'center',
  },
  hint: {
    backgroundColor: 'rgba(248,81,73,0.92)',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 10,
    marginBottom: 12,
  },
  hintText: {
    color: '#ffffff',
    fontSize: 14,
    textAlign: 'center',
  },
  status: {
    width: '100%',
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: 'rgba(13,17,23,0.82)',
    borderWidth: 1,
    borderColor: 'rgba(210,153,34,0.6)',
  },
  statusReady: {
    borderColor: 'rgba(63,185,80,0.85)',
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  statusText: {
    color: '#f0d58c',
    fontSize: 15,
    fontWeight: '600',
    textAlign: 'center',
    marginLeft: 8,
  },
  statusTextReady: {
    color: '#7ee787',
  },
  dots: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 8,
    gap: 6,
  },
  dot: {
    width: 22,
    height: 4,
    borderRadius: 2,
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  dotOn: {
    backgroundColor: '#3fb950',
  },
  footer: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 11,
    marginTop: 12,
    textAlign: 'center',
  },
  processing: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(3,7,12,0.82)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  processingText: {
    color: '#f0f6fc',
    fontSize: 17,
    fontWeight: '600',
    marginTop: 14,
  },
});
