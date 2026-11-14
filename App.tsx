import type {ScanResult} from './src/native/ExamScanner';
import ResultScreen from './src/screens/ResultScreen';
import ScannerScreen from './src/screens/ScannerScreen';
import React, {useState} from 'react';
import {StatusBar, StyleSheet, View} from 'react-native';
import {SafeAreaProvider} from 'react-native-safe-area-context';

function App() {
  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" backgroundColor="#000000" />
      <View style={styles.container}>
        <Root />
      </View>
    </SafeAreaProvider>
  );
}

function Root() {
  const [result, setResult] = useState<ScanResult | null>(null);

  if (result != null) {
    return (
      <ResultScreen result={result} onRetry={() => setResult(null)} />
    );
  }
  return <ScannerScreen onResult={setResult} />;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
});

export default App;
