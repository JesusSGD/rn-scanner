import { NativeModules } from 'react-native';

export type ScanAnswer = {
  number: number;
  letter: string;
  ratio: number;
  secondRatio: number;
  ambiguous: boolean;
};

export type ScanResult = {
  ok: boolean;
  reason: string;
  message: string;
  pageFound: boolean;
  gridFound: boolean;
  sharpness: number;
  brightness: number;
  glare: number;
  coverage: number;
  pageAreaRatio: number;
  orientation: number;
  /** Printed blocks (A-E groups) found on the sheet: 1 to 6. */
  blocks: number;
  elapsedMs: number;
  answers: ScanAnswer[];
  warnings: string[];
  debugPath?: string;
};

type ExamScannerNative = {
  scanPage(path: string): Promise<ScanResult>;
};

const nativeScanner = NativeModules.ExamScanner as
  | ExamScannerNative
  | undefined;

/** Reads the answer sheet photo at [path] completely on the device. */
export function scanPage(path: string): Promise<ScanResult> {
  if (!nativeScanner) {
    return Promise.reject(
      new Error('El módulo nativo del escáner no está disponible.'),
    );
  }
  return nativeScanner.scanPage(path);
}

export function answeredCount(result: ScanResult): number {
  return result.answers.filter(answer => answer.letter.length > 0).length;
}
