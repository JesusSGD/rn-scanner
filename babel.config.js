module.exports = {
  presets: ['@react-native/babel-preset'],
  plugins: [
    '@babel/plugin-proposal-optional-chaining',
    '@babel/plugin-transform-nullish-coalescing-operator',
    // Must be last: transforms 'worklet' functions for VisionCamera frame processors
    'react-native-worklets-core/plugin',
  ],
};
