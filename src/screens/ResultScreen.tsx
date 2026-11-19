import type {ScanResult} from '../native/ExamScanner';
import {answeredCount} from '../native/ExamScanner';
import React from 'react';
import {Image, ScrollView, StyleSheet, Text, TouchableOpacity, View} from 'react-native';

type Props = {
  result: ScanResult;
  onRetry: () => void;
};

const REASON_TITLES: Record<string, string> = {
  blurry: 'La foto salió movida',
  too_dark: 'Hay muy poca luz',
  too_bright: 'La hoja está sobreexpuesta',
  glare: 'Hay un reflejo sobre la hoja',
  no_pattern: 'No se reconoce la hoja',
  no_page: 'No se ve la hoja completa',
  page_too_small: 'La hoja está muy lejos',
  error: 'No se pudo leer la foto',
};

export default function ResultScreen({result, onRetry}: Props) {
  const answered = result.answers.filter(answer => answer.letter.length > 0);
  const total = answeredCount(result);
  const title = result.ok
    ? 'Hoja leída'
    : REASON_TITLES[result.reason] ?? 'No se pudo leer la hoja';

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={[styles.badge, result.ok ? styles.badgeOk : styles.badgeFail]}>
          <Text style={styles.badgeIcon}>{result.ok ? '✓' : '!'}</Text>
        </View>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.message}>{result.message}</Text>

        {result.ok ? (
          <>
            <View style={styles.summary}>
              <Text style={styles.summaryNumber}>{total}</Text>
              <Text style={styles.summaryLabel}>
                {total === 1 ? 'respuesta marcada' : 'respuestas marcadas'}
              </Text>
              {result.blocks > 0 ? (
                <Text style={styles.summaryDetail}>
                  {result.blocks}{' '}
                  {result.blocks === 1 ? 'bloque A-E' : 'bloques A-E'}
                </Text>
              ) : null}
            </View>

            {answered.length > 0 ? (
              <View style={styles.grid}>
                {answered.map(answer => (
                  <View
                    key={answer.number}
                    style={[
                      styles.answer,
                      answer.ambiguous && styles.answerAmbiguous,
                    ]}>
                    <Text style={styles.answerNumber}>{answer.number}</Text>
                    <Text style={styles.answerLetter}>{answer.letter}</Text>
                  </View>
                ))}
              </View>
            ) : (
              <Text style={styles.empty}>
                No se detectó ninguna burbuja rellena.
              </Text>
            )}
          </>
        ) : null}

        {result.debugPath ? (
          <View style={styles.debugSection}>
            <Text style={styles.debugLabel}>Vista de burbujas detectadas</Text>
            <Image
              source={{uri: 'file://' + result.debugPath}}
              style={styles.debugImage}
              resizeMode="contain"
            />
            <Text style={styles.debugHint}>
              Azul = cuadro de respuestas · Verde = marcada · Gris = vacía · Rojo = sin respuesta
            </Text>
          </View>
        ) : null}

        {result.warnings.length > 0 ? (
          <View style={styles.warnings}>
            {result.warnings.map(warning => (
              <Text key={warning} style={styles.warning}>
                • {warning}
              </Text>
            ))}
          </View>
        ) : null}
      </ScrollView>

      <TouchableOpacity style={styles.button} onPress={onRetry}>
        <Text style={styles.buttonText}>Escanear otra hoja</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0d1117',
  },
  content: {
    paddingHorizontal: 24,
    paddingTop: 56,
    paddingBottom: 32,
    alignItems: 'center',
  },
  badge: {
    width: 72,
    height: 72,
    borderRadius: 36,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  badgeOk: {
    backgroundColor: '#16351f',
    borderWidth: 2,
    borderColor: '#3fb950',
  },
  badgeFail: {
    backgroundColor: '#3a1d1d',
    borderWidth: 2,
    borderColor: '#f85149',
  },
  badgeIcon: {
    color: '#ffffff',
    fontSize: 34,
    fontWeight: '700',
  },
  title: {
    color: '#f0f6fc',
    fontSize: 26,
    fontWeight: '700',
    textAlign: 'center',
  },
  message: {
    color: '#9aa4af',
    fontSize: 15,
    lineHeight: 22,
    textAlign: 'center',
    marginTop: 10,
    marginBottom: 24,
  },
  summary: {
    alignItems: 'center',
    marginBottom: 20,
  },
  summaryNumber: {
    color: '#3fb950',
    fontSize: 52,
    fontWeight: '800',
    lineHeight: 56,
  },
  summaryLabel: {
    color: '#9aa4af',
    fontSize: 14,
  },
  summaryDetail: {
    color: '#6e7681',
    fontSize: 12,
    marginTop: 4,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 8,
  },
  answer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#161b22',
    borderWidth: 1,
    borderColor: '#30363d',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
    minWidth: 64,
    justifyContent: 'center',
  },
  answerAmbiguous: {
    borderColor: '#d29922',
  },
  answerNumber: {
    color: '#8b949e',
    fontSize: 13,
    marginRight: 8,
  },
  answerLetter: {
    color: '#f0f6fc',
    fontSize: 18,
    fontWeight: '700',
  },
  empty: {
    color: '#8b949e',
    fontSize: 15,
    textAlign: 'center',
  },
  warnings: {
    marginTop: 24,
    alignSelf: 'stretch',
  },
  warning: {
    color: '#d29922',
    fontSize: 13,
    lineHeight: 20,
  },
  button: {
    margin: 20,
    backgroundColor: '#238636',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 17,
    fontWeight: '700',
  },
  debugSection: {
    alignSelf: 'stretch',
    marginTop: 24,
    alignItems: 'center',
  },
  debugLabel: {
    color: '#8b949e',
    fontSize: 12,
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 0.6,
  },
  debugImage: {
    width: '100%',
    height: 320,
    borderRadius: 10,
    backgroundColor: '#161b22',
  },
  debugHint: {
    color: '#6e7681',
    fontSize: 11,
    marginTop: 6,
    textAlign: 'center',
  },
});
