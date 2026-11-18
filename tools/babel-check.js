// One-off probe: does the worklets babel plugin transform vision-camera's
// withFrameRefCounting (shipped pre-compiled in lib/commonjs)?
const babel = require('@babel/core');
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const file = path.join(
  root,
  'node_modules/react-native-vision-camera/lib/commonjs/frame-processors/withFrameRefCounting.js'
);
const src = fs.readFileSync(file, 'utf8');

const result = babel.transformSync(src, {
  filename: file,
  cwd: root,
  configFile: path.join(root, 'babel.config.js'),
  babelrc: false,
  compact: false,
  envName: 'production',
});
const out = result.code;
console.log('hasWorkletHash:', out.includes('__workletHash'));
console.log('hasInitData:', out.includes('__initData'));
console.log(out.slice(0, 1600));
