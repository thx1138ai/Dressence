const { getDefaultConfig } = require('expo/metro-config');
const { mergeConfig } = require('@react-native/metro-config');

const defaultConfig = getDefaultConfig(__dirname);
const { assetExts, sourceExts } = defaultConfig.resolver;

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('metro-config').MetroConfig}
 */
const config = {
  resolver: {
    assetExts: [...assetExts, 'tflite', 'bin', 'onnx', 'json'], // Keep TFJS and other ML extensions
    sourceExts: [...sourceExts],
  },
  // ------ START: COMMENT OUT / REMOVE THIS BLOCK ------
  // transformer: {
  //   getTransformOptions: async () => ({
  //     transform: {
  //       experimentalImportSupport: false,
  //       inlineRequires: true, // Temporarily disable this setting
  //     },
  //   }),
  // },
  // ------ END: COMMENT OUT / REMOVE THIS BLOCK ------
};

module.exports = mergeConfig(defaultConfig, config);
