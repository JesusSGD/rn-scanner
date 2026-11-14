/**
 * @format
 */

import React from 'react';
import ReactTestRenderer from 'react-test-renderer';

jest.mock('react-native-vision-camera', () => ({
  Camera: () => null,
  useCameraDevice: () => undefined,
  useCameraPermission: () => ({
    hasPermission: false,
    requestPermission: jest.fn(),
  }),
  useFrameProcessor: () => undefined,
  VisionCameraProxy: {initFrameProcessorPlugin: () => undefined},
  runAtTargetFps: (_fps: number, fn: () => unknown) => fn(),
}));

jest.mock('react-native-worklets-core', () => ({
  useSharedValue: (value: unknown) => ({value}),
  useRunOnJS: (fn: unknown) => fn,
}));

import App from '../App';

test('renders the camera permission gate', async () => {
  let tree: ReactTestRenderer.ReactTestRenderer | undefined;
  await ReactTestRenderer.act(() => {
    tree = ReactTestRenderer.create(<App />);
  });
  expect(tree).toBeDefined();
});
