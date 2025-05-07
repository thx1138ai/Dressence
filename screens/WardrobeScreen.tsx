/*
 * This component displays the user's wardrobe as a grid of clothing items,
 * provides functionality for adding new items via camera or photo library,
 * and runs ONNX-based AI pipelines for:
 *   1. Clothing detection & segmentation (YOLO + mask generation)
 *   2. Item classification (category, attributes, colors, PCA features)
 *   3. Context prediction (formality, temperature, seasons)
 *
 * Features:
 *  - Persistent storage: images and metadata are saved to AsyncStorage and the file system
 *  - Batch operations: long-press to select multiple items, delete selected items at once
 *  - Single-item detail modal: tap to view expanded image, edit category, see usage context,
 *    dominant colors, and similar-item recommendations based on PCA distance
 *  - Robust image preprocessing: nearest-neighbor resizing, padding, normalization for ONNX inputs
 *  - Mask-based segmentation: crops each detected item before classification
 *  - User feedback: processing overlays, status messages for errors (unsupported formats, no detection)
 *  - Modular ONNX sessions: separate inference sessions for detector, classifier, and context models
 *
 * Data flow:
 *  - On mount, loads saved items and initializes ONNX sessions
 *  - Adds new items through `takePhoto` or `selectPhoto`, classifies each, and updates wardrobe
 *  - Organises items into `FlatList` sections by category
 *  - Handles item selection, deletion, and category editing
 *  - Debug logging throughout for tensor shapes, PCA scaling, mask generation, and error tracing
 *
*/

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  TouchableOpacity,
  FlatList,
  Image,
  Dimensions,
  Alert,
  Platform,
  Modal,
  TouchableWithoutFeedback,
  ScrollView,
  ActivityIndicator
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import AsyncStorage from '@react-native-async-storage/async-storage';
// @ts-ignore
import * as ort from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import * as ImagePicker from 'react-native-image-picker';
import jpeg from 'jpeg-js';
import { Buffer } from 'buffer';
import contextScalars from './assets/context_scalar_params.json';
import pcaComponentsData from './assets/pca_components_with_weights.json';
import LinearGradient from 'react-native-linear-gradient';
import { PNG } from 'pngjs/browser';
global.Buffer = Buffer;

const FORMALITY_SCALE = ['Very Casual', 'Casual', 'Smart Casual', 'Business Casual', 'Formal'];
const TEMPERATURE_SCALE = ['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot']; // 0=Very Cold, 4=Hot
const SEASON_ICONS: { [key: string]: string } = {
  Spring: '🌸',
  Summer: '☀️',
  Fall: '🍂',
  Winter: '❄️',
};

type RootStackParamList = {
  Swipe: undefined;
  Wardrobe: undefined;
  OutfitGenerator: undefined;
};

type SwipeScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Swipe'>;

type Props = {
  navigation: SwipeScreenNavigationProp;
};

type WardrobeItem = {
  id: string;
  uri: string;
  processedUri: string;
  category?: string;
  className?: string;
  colors?: Array<[string, number, number[]]>;
  principalComponents?: number[];
  formality?: number;
  temperature?: number;
  seasons?: string[];
};

type PredictionResult = {
  category: string;
  attributes: { prob: number; index: number }[];
  attributeProbabilities?: number[];
  colors?: Array<[string, number, number[]]>;
  principalComponents?: number[];
  formality?: number;
  temperature?: number;
  seasons?: string[];
};

type CategorySection = {
  title: string;
  data: WardrobeItem[];
};

const deepfashion2_class_names: {[key: number]: string} = {
    0: "short_sleeve_top",
    1: "long_sleeve_top",
    2: "short_sleeve_outwear",
    3: "long_sleeve_outwear",
    4: "vest",
    5: "sling",
    6: "shorts",
    7: "trousers",
    8: "skirt",
    9: "short_sleeve_dress",
    10: "long_sleeve_dress",
    11: "vest_dress",
    12: "sling_dress"
};

const category_names = [
  'T-shirts/Tanks',
  'Shirts',
  'Sweaters/Hoodies',
  'Light Outerwear',
  'Heavy Outerwear',
  'Trousers',
  'Shorts',
  'Skirts',
  'Athletic Bottoms',
  'Dresses',
  'Jumpsuits',
  'Formal Wear',
  'Misc',
];

const COLOR_MAP: { [key: string]: string } = {
  'white': '#FFFFFF',
  'black': '#000000',
  'red': '#FF0000',
  'green': '#008000',
  'blue': '#0000FF',
  'yellow': '#FFFF00',
  'cyan': '#00FFFF',
  'magenta': '#FF00FF',
  'silver': '#C0C0C0',
  'gray': '#808080',
  'maroon': '#800000',
  'olive': '#808000',
  'purple': '#800080',
  'teal': '#008080',
  'navy': '#000080',
  'orange': '#FFA500',
  'alice_blue': '#F0F8FF',
  'antique_white': '#FAEBDA',
  'aqua': '#00FFFF',
  'aquamarine': '#7FFFD4',
  'azure': '#F0FFFF',
  'beige': '#F5F5DC',
  'bisque': '#FFE4C4',
  'blanched_almond': '#FFEBCD',
  'blue_violet': '#8A2BE2',
  'brown': '#A52A2A',
  'burly_wood': '#DEB887',
  'cadet_blue': '#5F9EA0',
  'chartreuse': '#7FFF00',
  'chocolate': '#D2691E',
  'coral': '#FF7F50',
  'cornflower_blue': '#6495ED',
  'cornsilk': '#FFF8DC',
  'crimson': '#DC143C',
  'dark_blue': '#00008B',
  'dark_cyan': '#008B8B',
  'dark_goldenrod': '#B8860B',
  'dark_gray': '#A9A9A9',
  'dark_green': '#006400',
  'dark_khaki': '#BDB76B',
  'dark_magenta': '#8B008B',
  'dark_olive_green': '#556B2F',
  'dark_orange': '#FF8C00',
  'dark_orchid': '#9932CC',
  'dark_red': '#8B0000',
  'dark_salmon': '#E9967A',
  'dark_sea_green': '#8FBC8F',
  'dark_slate_blue': '#483D8B',
  'dark_slate_gray': '#2F4F4F',
  'dark_turquoise': '#00CED1',
  'dark_violet': '#9400D3',
  'deep_pink': '#FF1493',
  'deep_sky_blue': '#00BFFF',
  'dim_gray': '#696969',
  'dodger_blue': '#1E90FF',
  'firebrick': '#B22222',
  'floral_white': '#FFFAF0',
  'forest_green': '#228B22',
  'fuchsia': '#FF00FF',
  'gainsboro': '#DCDCDC',
  'ghost_white': '#F8F8FF',
  'gold': '#FFD700',
  'goldenrod': '#DAA520',
  'green_yellow': '#ADFF2F',
  'honeydew': '#F0FFF0',
  'hot_pink': '#FF69B4',
  'indian_red': '#CD5C5C',
  'indigo': '#4B0082',
  'ivory': '#FFFFF0',
  'khaki': '#F0E68C',
  'lavender': '#E6E6FA',
  'lavender_blush': '#FFF0F5',
  'lawn_green': '#7CFC00',
  'lemon_chiffon': '#FFFACD',
  'light_blue': '#ADD8E6',
};

const formality_levels = ['Very Casual', 'Casual', 'Smart Casual', 'Business Casual', 'Formal'];
const temperature_levels = ['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot'];
const season_names = ['Spring', 'Summer', 'Fall', 'Winter'];

const formatColorName = (colorName: string): string => {
  return colorName
    .split('_')
    .map((word, index) => 
      index === 0 
        ? word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
        : word.toLowerCase()
    )
    .join(' ');
};

function debugTensorData(tensorData: Float32Array) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let hasNaN = false;

  for (let i = 0; i < tensorData.length; i++) {
    const value = tensorData[i];
    if (value < min) min = value;
    if (value > max) max = value;
    if (Number.isNaN(value)) hasNaN = true;
    sum += value;
  }

  const mean = sum / tensorData.length;
  return { min, max, mean, hasNaN };
}

async function getModelPath(modelType: string) {
    let modelFileName: string;
    switch (modelType) {
        case 'classifier':
            modelFileName = 'clothing_classifier.onnx';
            break;
        case 'detector':
            modelFileName = 'clothing_detector.onnx';
            break;
        case 'context':
            modelFileName = 'clothing_context_predictor.onnx';
            break;
        default:
            throw new Error(`Unknown model type: ${modelType}`);
    }

    let modelPath: string;
    if (Platform.OS === 'android') {
        modelPath = `${RNFS.DocumentDirectoryPath}/${modelFileName}`;
        try {
            const exists = await RNFS.exists(modelPath);
            if (!exists) {
                console.log(`Copying ${modelFileName} from assets to ${modelPath}`);
                // Copy from assets directory on Android (for future)
                await RNFS.copyFileAssets(`screens/assets/${modelFileName}`, modelPath);
            }
            console.log(`[getModelPath - Android] Using model at: ${modelPath}`);
            return modelPath;
        } catch (error) {
            console.error(`Error preparing model ${modelFileName}:`, error);
            throw error;
        }
    } else { // iOS
        modelPath = `${RNFS.MainBundlePath}/${modelFileName}`;
        console.log(`[getModelPath - iOS] Using model at: ${modelPath}`);
        const exists = await RNFS.exists(modelPath);
        if (!exists) {
            console.warn(`Model file ${modelFileName} does not exist at path: ${modelPath}. Ensure it's added to the main bundle.`);
        }
        return modelPath;
    }
}

function softmax(logits: Float32Array | number[]): Float32Array {
  const logitsArray = Float32Array.from(logits);
  const maxLogit = Math.max(...logitsArray);
  const exps = logitsArray.map(x => Math.exp(x - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  if (sum === 0) {
      return new Float32Array(logitsArray.length).fill(1 / logitsArray.length);
  }
  return new Float32Array(exps.map(exp => exp / sum));
}

async function prepareImageForDetector(imageUri: string) {
    try {
      // Check if the file is a JPEG or PNG
      if (!imageUri.match(/\.(jpe?g|png)$/i)) {
        console.log('[prepareImageForDetector] Unsupported format:', imageUri);
        throw new Error('unsupported_image_format');
      }

      const imageData = await RNFS.readFile(imageUri, 'base64');
      const buffer = Buffer.from(imageData, 'base64');
      
      // Decode JPEG or PNG into raw RGB(A)
      let rawImg;
      try {
        if (imageUri.toLowerCase().endsWith('.png')) {
          rawImg = PNG.sync.read(buffer);
        } else {
          rawImg = jpeg.decode(buffer, { useTArray: true, formatAsRGBA: true });
        }
      } catch (err) {
        console.log('[prepareImageForDetector] Skipping image due to decode issue');
        throw new Error('unsupported_image_format');
      }
      
      console.log(`Original image dimensions: ${rawImg.width}x${rawImg.height}`);
      
      const maxDim = Math.max(rawImg.width, rawImg.height);
      
      const squareCanvasData = new Uint8Array(maxDim * maxDim * 3);
      squareCanvasData.fill(255);
      
      const offsetX = Math.floor((maxDim - rawImg.width) / 2);
      const offsetY = Math.floor((maxDim - rawImg.height) / 2);
      
      const channels = rawImg.data.length / (rawImg.width * rawImg.height); // should be 4
      
      // Copy the original image onto the square canvas
      for (let y = 0; y < rawImg.height; y++) {
        for (let x = 0; x < rawImg.width; x++) {
          const srcPos = (y * rawImg.width + x) * channels;
          const destPos = ((y + offsetY) * maxDim + (x + offsetX)) * 3;
          
          squareCanvasData[destPos] = rawImg.data[srcPos];         // R
          squareCanvasData[destPos + 1] = rawImg.data[srcPos + 1]; // G
          squareCanvasData[destPos + 2] = rawImg.data[srcPos + 2]; // B
        }
      }
      
      // Now resize the square canvas to TARGET_SIZE (640x640)
      const TARGET_SIZE = 640;
      const resizedData = new Uint8Array(TARGET_SIZE * TARGET_SIZE * 3);
      const scale = maxDim / TARGET_SIZE;
      
      // Simple nearest-neighbor resize 
      for (let y = 0; y < TARGET_SIZE; y++) {
        for (let x = 0; x < TARGET_SIZE; x++) {
          const srcX = Math.min(Math.floor(x * scale), maxDim - 1);
          const srcY = Math.min(Math.floor(y * scale), maxDim - 1);
          const srcPos = (srcY * maxDim + srcX) * 3;
          const destPos = (y * TARGET_SIZE + x) * 3;
          
          resizedData[destPos] = squareCanvasData[srcPos];         // R
          resizedData[destPos + 1] = squareCanvasData[srcPos + 1]; // G
          resizedData[destPos + 2] = squareCanvasData[srcPos + 2]; // B
        }
      }
      
      console.log(`Resized to: ${TARGET_SIZE}x${TARGET_SIZE}`);
      
      // Convert to float32 and normalise to [0,1] range
      const tensorData = new Float32Array(3 * TARGET_SIZE * TARGET_SIZE);
      const channelSize = TARGET_SIZE * TARGET_SIZE;
      
      for (let y = 0; y < TARGET_SIZE; y++) {
        for (let x = 0; x < TARGET_SIZE; x++) {
          const pos = (y * TARGET_SIZE + x) * 3;
          const idx = y * TARGET_SIZE + x;
          
          tensorData[idx] = resizedData[pos] / 255.0;                    // R
          tensorData[channelSize + idx] = resizedData[pos + 1] / 255.0;  // G
          tensorData[2 * channelSize + idx] = resizedData[pos + 2] / 255.0; // B
        }
      }
      
      console.log('Tensor data prepared with normalization');
      
      return new ort.Tensor('float32', tensorData, [1, 3, TARGET_SIZE, TARGET_SIZE]);
      
    } catch (error) {
      console.log('Image preparation skipped:', error);
      throw error;
    }
  }

  // Function to process YOLO model output
  function processYoloOutput(outputs: Record<string, ort.Tensor>) {
    try {
      console.log('Processing YOLO outputs');
      
      // Extract the detection output tensor - shape [1, 49, 8400]
      const detectionOutput = outputs.output0 || outputs[Object.keys(outputs)[0]];
      const outputData = detectionOutput.data as Float32Array;
      const outputShape = detectionOutput.dims;
      
      // Extract the mask prototypes tensor - shape [1, 32, 160, 160]
      const protoOutput = outputs.output1 || outputs[Object.keys(outputs)[1]];
      const protoData = protoOutput.data as Float32Array;
      const protoShape = protoOutput.dims;
      
      console.log('Detection output shape:', outputShape);
      console.log('Proto output shape:', protoShape);
      
      if (outputShape.length !== 3 || protoShape.length !== 4) {
        console.error('Unexpected output shapes');
        return [];
      }
      
      const numClasses = 13;
      const boxDim = 4; // x, y, w, h
      const maskCoeffDim = 32;
      const numDetections = outputShape[2]; // 8400 possible detections
      
      // Process each detection
      const detections: any[] = [];
      const threshold = 0.5; // Confidence threshold
      
      for (let i = 0; i < numDetections; i++) {
        // Extract class probabilities (starts after bbox values)
        const classProbs = Array(numClasses);
        for (let c = 0; c < numClasses; c++) {
          // Index starts at boxDim (4) + c
          classProbs[c] = outputData[(boxDim + c) * numDetections + i];
        }
        
        // Find highest class probability
        const maxClassProb = Math.max(...classProbs);
        const classIdx = classProbs.indexOf(maxClassProb);
        
        // Use class probability as confidence
        const confidence = maxClassProb;
        
        // If confidence is above threshold, extract additional data
        if (confidence > threshold) {
          // Extract bounding box
          const x = outputData[0 * numDetections + i];
          const y = outputData[1 * numDetections + i];
          const w = outputData[2 * numDetections + i];
          const h = outputData[3 * numDetections + i];
          
          // Extract mask coefficients - starts after box and class values
          const maskCoeffs = new Float32Array(maskCoeffDim);
          for (let m = 0; m < maskCoeffDim; m++) {
            // Index starts at boxDim (4) + numClasses (13) + m
            const coeffIndex = (boxDim + numClasses + m) * numDetections + i;
            maskCoeffs[m] = outputData[coeffIndex];
          }
          
          // Check the last coefficient to verify our indexing is correct
          if (isNaN(maskCoeffs[maskCoeffDim - 1])) {
            console.warn(`NaN detected in last coeff (index ${maskCoeffDim - 1}) for detection ${i} WITH CORRECTED INDEXING`);
          }
          
          // Generate the raw mask
          const protoH = protoShape[2]; // 160
          const protoW = protoShape[3]; // 160
          const rawMask = generateMask(maskCoeffs, protoData, protoH, protoW);
          
          detections.push({
            classIdx,
            className: deepfashion2_class_names[classIdx],
            confidence,
            bbox: [x, y, w, h], // center format [x, y, w, h]
            bboxCorners: [
              x - w/2, 
              y - h/2, 
              x + w/2, 
              y + h/2
            ],
            allClassProbs: [...classProbs],
            maskCoefficients: Array.from(maskCoeffs),
            rawMask: rawMask,
            detectionIndex: i
          });
        }
      }
      
      console.log(`Found ${detections.length} detections above threshold ${threshold}`);
      
      // Sort by confidence (highest first)
      detections.sort((a, b) => b.confidence - a.confidence);
      
      const nmsResults = applyNms(detections);
      console.log(`After NMS: ${nmsResults.length} detections`);
      
      return nmsResults;
    } catch (error) {
      console.error('Error processing YOLO output:', error);
      return [];
    }
  }
  
  // Calculate IoU (Intersection over Union) for NMS
  function calculateIou(box1: number[], box2: number[]) {
    // Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]
    const box1X1 = box1[0] - box1[2]/2;
    const box1Y1 = box1[1] - box1[3]/2;
    const box1X2 = box1[0] + box1[2]/2;
    const box1Y2 = box1[1] + box1[3]/2;
    
    const box2X1 = box2[0] - box2[2]/2;
    const box2Y1 = box2[1] - box2[3]/2;
    const box2X2 = box2[0] + box2[2]/2;
    const box2Y2 = box2[1] + box2[3]/2;
    
    // Calculate intersection area
    const xLeft = Math.max(box1X1, box2X1);
    const yTop = Math.max(box1Y1, box2Y1);
    const xRight = Math.min(box1X2, box2X2);
    const yBottom = Math.min(box1Y2, box2Y2);
    
    if (xRight < xLeft || yBottom < yTop) {
      return 0; // So no intersection
    }
    
    const intersection = (xRight - xLeft) * (yBottom - yTop);
    
    // Calculate union area
    const box1Area = (box1X2 - box1X1) * (box1Y2 - box1Y1);
    const box2Area = (box2X2 - box2X1) * (box2Y2 - box2Y1);
    const union = box1Area + box2Area - intersection;
    
    return intersection / union;
  }
  
  // Apply Non-Maximum Suppression
  function applyNms(detections: any[], iouThreshold = 0.5) {
    const result: any[] = [];
    
    // Process each class separately
    for (let c = 0; c < 13; c++) {
      // Get detections for this class
      const classDetections = detections.filter(d => d.classIdx === c);
      
      // Process while we have detections
      while (classDetections.length > 0) {
        // Take detection with highest confidence
        const bestDetection = classDetections.shift()!;
        result.push(bestDetection);
        
        // Filter out overlapping detections
        let i = 0;
        while (i < classDetections.length) {
          const iou = calculateIou(bestDetection.bbox, classDetections[i].bbox);
          if (iou > iouThreshold) {
            classDetections.splice(i, 1);
          } else {
            i++;
          }
        }
      }
    }
    
    // Sort results by confidence
    return result.sort((a, b) => b.confidence - a.confidence);
  }

async function prepareImageForOnnx(imageUri: string): Promise<ort.Tensor> {
  console.log('[prepareImageForOnnx] Reading image:', imageUri);
  try {
    // Check if the file is a JPEG or PNG
    if (!imageUri.match(/\.(jpe?g|png)$/i)) {
      console.log('[prepareImageForOnnx] Unsupported format:', imageUri);
      throw new Error('unsupported_image_format');
    }
    
    // Read the image file as Base64 and create a buffer
    const imageData = await RNFS.readFile(imageUri, 'base64');
    const buffer = Buffer.from(imageData, 'base64');

    // Decode the image data
    let rawImg: { data: Uint8Array; width: number; height: number };
    try {
      if (imageUri.toLowerCase().endsWith('.png')) {
        rawImg = PNG.sync.read(buffer);
      } else {
        rawImg = jpeg.decode(buffer, { useTArray: true, formatAsRGBA: true });
      }
    } catch (err) {
      console.log('[prepareImageForOnnx] Skipping image due to decode issue');
      throw new Error('unsupported_image_format');
    }
    if (!rawImg || !rawImg.data || !rawImg.width || !rawImg.height) {
      throw new Error('Invalid image data after decoding');
    }
    console.log(`Original dimensions: ${rawImg.width}x${rawImg.height}`);
    
    const TARGET_SIZE = 224;
    // Create a canvas (flat array) for a 224×224 image with 3 channels and fill with 255 (white)
    const canvasData = new Uint8Array(TARGET_SIZE * TARGET_SIZE * 3);
    canvasData.fill(255);

    // Compute scale to fit the image into TARGET_SIZE while preserving aspect ratio
    const scale = Math.min(TARGET_SIZE / rawImg.width, TARGET_SIZE / rawImg.height);
    const newWidth = Math.round(rawImg.width * scale);
    const newHeight = Math.round(rawImg.height * scale);
    
    // Nearest neighbor resize into a temporary buffer
    const resizedData = new Uint8Array(newWidth * newHeight * 3);
    
    // rawImg.data is RGBA (4 channels) for PNG or JPEG
    const channels = rawImg.data.length / (rawImg.width * rawImg.height); // should be 4
    for (let y = 0; y < newHeight; y++) {
      for (let x = 0; x < newWidth; x++) {
        const srcX = Math.floor(x / scale);
        const srcY = Math.floor(y / scale);
        const srcBase = (srcY * rawImg.width + srcX) * channels;
        const dstBase = (y * newWidth + x) * 3;
        resizedData[dstBase]     = rawImg.data[srcBase + 0]; // R
        resizedData[dstBase + 1] = rawImg.data[srcBase + 1]; // G
        resizedData[dstBase + 2] = rawImg.data[srcBase + 2]; // B
      }
    }
    
    // Figure out padding to center the resized image in a 224×224 canvas.
    const padX = Math.floor((TARGET_SIZE - newWidth) / 2);
    const padY = Math.floor((TARGET_SIZE - newHeight) / 2);
    
    // Copy resized image data into the canvas at the computed offset.
    for (let y = 0; y < newHeight; y++) {
      for (let x = 0; x < newWidth; x++) {
        const srcIdx = (y * newWidth + x) * 3;
        const dstX = x + padX;
        const dstY = y + padY;
        const dstIdx = (dstY * TARGET_SIZE + dstX) * 3;
        canvasData[dstIdx] = resizedData[srcIdx];
        canvasData[dstIdx + 1] = resizedData[srcIdx + 1];
        canvasData[dstIdx + 2] = resizedData[srcIdx + 2];
      }
    }
    
    // Convert the canvas data (HWC) to a Float32Array in CHW format.
    const tensorData = new Float32Array(3 * TARGET_SIZE * TARGET_SIZE);
    const channelSize = TARGET_SIZE * TARGET_SIZE;
    for (let y = 0; y < TARGET_SIZE; y++) {
      for (let x = 0; x < TARGET_SIZE; x++) {
        const pos = (y * TARGET_SIZE + x) * 3;
        const r = canvasData[pos];
        const g = canvasData[pos + 1];
        const b = canvasData[pos + 2];
        const idx = y * TARGET_SIZE + x;
        tensorData[idx] = r;
        tensorData[channelSize + idx] = g;
        tensorData[2 * channelSize + idx] = b;
      }
    }
  
    console.log(`[prepareImageForOnnx] Prepared tensor with shape [1, 3, ${TARGET_SIZE}, ${TARGET_SIZE}]`);
    return new ort.Tensor('float32', tensorData, [1, 3, TARGET_SIZE, TARGET_SIZE]);
  } catch (error) {
    console.error('[prepareImageForOnnx] Error:', error);
    throw error;
  }
}

const PERSISTENT_IMAGE_DIR = Platform.OS === 'ios' 
  ? `${RNFS.LibraryDirectoryPath}/persistent_wardrobe_images`
  : `${RNFS.ExternalDirectoryPath}/persistent_wardrobe_images`;


function decodeColorResults(colorResults: Float32Array[]): Array<[string, number, number[]]> {
    const colors: Array<[string, number, number[]]> = [];
      
    for (const result of colorResults) {
      const r = Math.round(result[0]);
      const g = Math.round(result[1]);
      const b = Math.round(result[2]);
      const percentage = result[3] / 100;
      
      const rgb = [r, g, b];
      const colorName = findClosestNamedColor(rgb);
      
      colors.push([colorName, percentage, rgb]);
    }
      
    console.log('Decoded colors:', colors);
    return colors;
}

function findClosestNamedColor(rgb: number[]): string {
    let closestColor = 'unknown';
    let minDistance = Number.MAX_VALUE;
    
    for (const [name, hex] of Object.entries(COLOR_MAP)) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        
        const distance = Math.sqrt(
            Math.pow(rgb[0] - r, 2) + 
            Math.pow(rgb[1] - g, 2) + 
            Math.pow(rgb[2] - b, 2)
        );
        
        if (distance < minDistance) {
            minDistance = distance;
            closestColor = name;
        }
    }
    
    return closestColor;
}

type Detection = {
  classIdx: number;
  className: string;
  confidence: number;
  bbox: number[];
  bboxCorners: number[];
  rawMask: Float32Array;
};

function resizeImageNearestNeighbor(
    srcData: Uint8Array, 
    srcW: number, srcH: number, 
    destW: number, destH: number
): Uint8Array {
    const resized = new Uint8Array(destW * destH * 3);
    const scaleX = srcW / destW;
    const scaleY = srcH / destH;
    for (let y = 0; y < destH; y++) {
        for (let x = 0; x < destW; x++) {
            const srcX = Math.min(Math.floor(x * scaleX), srcW - 1);
            const srcY = Math.min(Math.floor(y * scaleY), srcH - 1);
            const srcPos = (srcY * srcW + srcX) * 3;
            const destPos = (y * destW + x) * 3;
            resized[destPos] = srcData[srcPos];     // R
            resized[destPos + 1] = srcData[srcPos + 1]; // G
            resized[destPos + 2] = srcData[srcPos + 2]; // B
        }
    }
    return resized;
}

function resizeMaskBilinear(
    mask: Float32Array,
    srcW: number, srcH: number,
    destW: number, destH: number
): Float32Array {
    const resized = new Float32Array(destW * destH);
    const scaleX = srcW / destW;
    const scaleY = srcH / destH;

    const getPixel = (x: number, y: number): number => {
        // Clamp coordinates to be within source bounds
        const clampedX = Math.max(0, Math.min(srcW - 1, x));
        const clampedY = Math.max(0, Math.min(srcH - 1, y));
        return mask[clampedY * srcW + clampedX];
    };

    for (let yDest = 0; yDest < destH; yDest++) {
        for (let xDest = 0; xDest < destW; xDest++) {
            // Calculate corresponding fractional source coordinates
            const srcX_f = xDest * scaleX;
            const srcY_f = yDest * scaleY;

            // Get integer coordinates of the top-left source pixel
            const x1 = Math.floor(srcX_f);
            const y1 = Math.floor(srcY_f);

            // Calculate fractional parts (weights)
            const xFrac = srcX_f - x1;
            const yFrac = srcY_f - y1;

            // Get values of the 4 neighboring source pixels
            const p11 = getPixel(x1, y1);         // Top-left
            const p21 = getPixel(x1 + 1, y1);     // Top-right
            const p12 = getPixel(x1, y1 + 1);     // Bottom-left
            const p22 = getPixel(x1 + 1, y1 + 1); // Bottom-right

            // Interpolate horizontally
            const interpTop = p11 * (1 - xFrac) + p21 * xFrac;
            const interpBottom = p12 * (1 - xFrac) + p22 * xFrac;

            // Interpolate vertically
            const finalValue = interpTop * (1 - yFrac) + interpBottom * yFrac;

            resized[yDest * destW + xDest] = finalValue;
        }
    }
    return resized;
}

function calculatePcaScores(attributeProbs: number[]): number[] {
    const pcaScores: number[] = [];
    
    Object.keys(pcaComponentsData).forEach(componentKey => {
        const componentWeights = pcaComponentsData[componentKey as keyof typeof pcaComponentsData];
      let score = 0;
      
      Object.entries(componentWeights).forEach(([attributeKey, weight]) => {
        // Parse the attribute index from the key ('Attribute_123' -> 123)
        const attrIndex = parseInt(attributeKey.split('_')[1], 10);
        
        if (!isNaN(attrIndex) && attrIndex < attributeProbs.length) {
          score += attributeProbs[attrIndex] * (weight as number);
        }
      });
      
      pcaScores.push(score);
    });
    
    return pcaScores;
  }

async function segmentImageWithMask(originalImageUri: string, detection: Detection): Promise<string> {
  console.log('Starting segmentation process...');
  const DETECTOR_SIZE = 640;
  const CLASSIFIER_SIZE = 512;
  const MASK_SIZE = 160;

  try {
    // === Step 1: Reload and Prepare Detector Input Image Data ===
    console.log('Step 1: Preparing 640x640 image data...');
    const imageData = await RNFS.readFile(originalImageUri, 'base64');
    const buffer = Buffer.from(imageData, 'base64');
    
    let rawImg: { data: Uint8Array; width: number; height: number };
    if (originalImageUri.toLowerCase().endsWith('.png')) {
      // use pngjs to decode
      rawImg = PNG.sync.read(buffer);
    } else {
      rawImg = jpeg.decode(buffer, { useTArray: true, formatAsRGBA: true });
    }
    
    const maxDimOrig = Math.max(rawImg.width, rawImg.height);
    const squareCanvasData = new Uint8Array(maxDimOrig * maxDimOrig * 3).fill(255);
    const offsetX = Math.floor((maxDimOrig - rawImg.width) / 2);
    const offsetY = Math.floor((maxDimOrig - rawImg.height) / 2);
    const channels = rawImg.data.length / (rawImg.width * rawImg.height);
    
    for (let y = 0; y < rawImg.height; y++) {
      for (let x = 0; x < rawImg.width; x++) {
        const srcPos = (y * rawImg.width + x) * channels;
        const destPos = ((y + offsetY) * maxDimOrig + (x + offsetX)) * 3;
        squareCanvasData[destPos] = rawImg.data[srcPos];         // R
        squareCanvasData[destPos + 1] = rawImg.data[srcPos + 1]; // G
        squareCanvasData[destPos + 2] = rawImg.data[srcPos + 2]; // B
      }
    }
    const detectorInputImageData = resizeImageNearestNeighbor(squareCanvasData, maxDimOrig, maxDimOrig, DETECTOR_SIZE, DETECTOR_SIZE);
    console.log('Step 1: Done.');

    // Step 2: Resize rawMask (160x160) to 640x640
    console.log('Step 2: Resizing raw mask...');
    const resizedProbMask = resizeMaskBilinear(detection.rawMask, MASK_SIZE, MASK_SIZE, DETECTOR_SIZE, DETECTOR_SIZE);
    console.log('Step 2: Done.');

    // Step 3: Create Bounding Box Mask (640x640)
    console.log('Step 3: Creating bounding box mask...');
    const bboxMaskArray = new Float32Array(DETECTOR_SIZE * DETECTOR_SIZE).fill(0.0);
    // Ensure bboxCorners are integers and clamped
    const x1 = Math.max(0, Math.min(DETECTOR_SIZE - 1, Math.round(detection.bboxCorners[0])));
    const y1 = Math.max(0, Math.min(DETECTOR_SIZE - 1, Math.round(detection.bboxCorners[1])));
    const x2 = Math.max(0, Math.min(DETECTOR_SIZE - 1, Math.round(detection.bboxCorners[2])));
    const y2 = Math.max(0, Math.min(DETECTOR_SIZE - 1, Math.round(detection.bboxCorners[3])));
    for (let y = y1; y < y2; y++) {
      for (let x = x1; x < x2; x++) {
        bboxMaskArray[y * DETECTOR_SIZE + x] = 1.0;
      }
    }
    console.log('Step 3: Done.');

    // Step 4: Crop Probability Mask
    console.log('Step 4: Cropping probability mask...');
    const croppedProbMask = new Float32Array(DETECTOR_SIZE * DETECTOR_SIZE);
    for(let i=0; i < croppedProbMask.length; i++) {
        croppedProbMask[i] = resizedProbMask[i] * bboxMaskArray[i];
    }
    console.log('Step 4: Done.');

    // Step 5: Threshold Mask
    console.log('Step 5: Thresholding mask...');
    const binaryMask = new Uint8Array(DETECTOR_SIZE * DETECTOR_SIZE);
    for(let i=0; i < binaryMask.length; i++) {
        binaryMask[i] = croppedProbMask[i] > 0.5 ? 1 : 0;
    }
    console.log('Step 5: Done.');

    // Step 6: Apply Mask to Image Data
    console.log('Step 6: Applying mask to image...');
    const maskedImageData = new Uint8Array(DETECTOR_SIZE * DETECTOR_SIZE * 3).fill(255);
    for (let y = 0; y < DETECTOR_SIZE; y++) {
      for (let x = 0; x < DETECTOR_SIZE; x++) {
        const maskIdx = y * DETECTOR_SIZE + x;
        if (binaryMask[maskIdx] === 1) {
          const imgPos = maskIdx * 3;
          maskedImageData[imgPos] = detectorInputImageData[imgPos];     // R
          maskedImageData[imgPos + 1] = detectorInputImageData[imgPos + 1]; // G
          maskedImageData[imgPos + 2] = detectorInputImageData[imgPos + 2]; // B
        }
      }
    }
    console.log('Step 6: Done.');

    // Step 7: Crop Item using BBox (with Padding)
    console.log('Step 7: Cropping masked item...');
    const padding = 10; // Add some padding
    const cropX1 = Math.max(0, x1 - padding);
    const cropY1 = Math.max(0, y1 - padding);
    const cropX2 = Math.min(DETECTOR_SIZE, x2 + padding);
    const cropY2 = Math.min(DETECTOR_SIZE, y2 + padding);
    const cropWidth = cropX2 - cropX1;
    const cropHeight = cropY2 - cropY1;

    if (cropWidth <= 0 || cropHeight <= 0) {
        throw new Error(`Invalid crop dimensions: ${cropWidth}x${cropHeight}`);
    }

    const croppedItemData = new Uint8Array(cropWidth * cropHeight * 3);
    for (let y = 0; y < cropHeight; y++) {
      for (let x = 0; x < cropWidth; x++) {
        const srcX = cropX1 + x;
        const srcY = cropY1 + y;
        const srcPos = (srcY * DETECTOR_SIZE + srcX) * 3;
        const destPos = (y * cropWidth + x) * 3;
        croppedItemData[destPos] = maskedImageData[srcPos];
        croppedItemData[destPos + 1] = maskedImageData[srcPos + 1];
        croppedItemData[destPos + 2] = maskedImageData[srcPos + 2];
      }
    }
    console.log('Step 7: Done.');

    // Step 8: Pad Cropped Item to Square
    console.log('Step 8: Padding cropped item to square...');
    const maxCropDim = Math.max(cropWidth, cropHeight);
    const squareItemData = new Uint8Array(maxCropDim * maxCropDim * 3).fill(255);
    const padX = Math.floor((maxCropDim - cropWidth) / 2);
    const padY = Math.floor((maxCropDim - cropHeight) / 2);
    for (let y = 0; y < cropHeight; y++) {
      for (let x = 0; x < cropWidth; x++) {
        const srcPos = (y * cropWidth + x) * 3;
        const destX = padX + x;
        const destY = padY + y;
        const destPos = (destY * maxCropDim + destX) * 3;
        squareItemData[destPos] = croppedItemData[srcPos];
        squareItemData[destPos + 1] = croppedItemData[srcPos + 1];
        squareItemData[destPos + 2] = croppedItemData[srcPos + 2];
      }
    }
    console.log('Step 8: Done.');

    // Step 9: Resize Square Item for Classifier
    console.log(`Step 9: Resizing square item to ${CLASSIFIER_SIZE}x${CLASSIFIER_SIZE}...`);
    const finalItemData = resizeImageNearestNeighbor(squareItemData, maxCropDim, maxCropDim, CLASSIFIER_SIZE, CLASSIFIER_SIZE);
    console.log('Step 9: Done.');

    // Step 10: Save Final Image
    console.log('Step 10: Saving final segmented image...');
    // Convert RGB to RGBA for image encoding
    const finalItemDataRGBA = new Uint8Array(CLASSIFIER_SIZE * CLASSIFIER_SIZE * 4);
    for (let i = 0; i < CLASSIFIER_SIZE * CLASSIFIER_SIZE; i++) {
        finalItemDataRGBA[i * 4 + 0] = finalItemData[i * 3 + 0]; // R
        finalItemDataRGBA[i * 4 + 1] = finalItemData[i * 3 + 1]; // G
        finalItemDataRGBA[i * 4 + 2] = finalItemData[i * 3 + 2]; // B
        finalItemDataRGBA[i * 4 + 3] = 255;                      // A
    }
    
    // Determine file extension from original image
    const extMatch = originalImageUri.match(/\.(jpe?g|png)$/i);
    const ext = extMatch ? extMatch[1].toLowerCase() : 'jpg';
    
    await RNFS.mkdir(PERSISTENT_IMAGE_DIR);
    const timestamp = Date.now();
    const randomString = Math.random().toString(36).substring(7);
    const filename = `segmented_${timestamp}_${randomString}.${ext}`;
    const destPath = `${PERSISTENT_IMAGE_DIR}/${filename}`;
    
    const rawImageData = {
      data: finalItemDataRGBA,
      width: CLASSIFIER_SIZE,
      height: CLASSIFIER_SIZE,
    };
    
    // Encode based on file extension
    let encodedImageData: Buffer;
    if (ext === 'png') {
      // Use PNG encoder
      const pngImage = new PNG({
        width: CLASSIFIER_SIZE,
        height: CLASSIFIER_SIZE,
        inputHasAlpha: true
      });
      pngImage.data = Buffer.from(finalItemDataRGBA);
      encodedImageData = PNG.sync.write(pngImage);
    } else {
      const jpegImageData = jpeg.encode(rawImageData, 90); // Quality 90
      encodedImageData = jpegImageData.data;
    }
    
    await RNFS.writeFile(destPath, encodedImageData.toString('base64'), 'base64');
    console.log('Step 10: Saved segmented image to:', destPath);

    // Step 11: Return URI
    return destPath;

  } catch (error) {
    console.error('Error in segmentImageWithMask:', error);
    // Fallback: return original URI if segmentation fails
    return originalImageUri; 
  }
}

async function classifyImage(
  imageUri: string,
  detectorSess: ort.InferenceSession,
  classificationSess: ort.InferenceSession,
  contextSess: ort.InferenceSession
): Promise<{ prediction: PredictionResult; processedUri: string; className: string } | { error: string }> {
    try {
        console.log("--- Starting Clothing Detection ---");
        // Step 1: Detection & Segmentation
        const tensor = await prepareImageForDetector(imageUri);
        const inputName = detectorSess.inputNames[0];
        const feeds: { [key: string]: ort.Tensor } = { [inputName]: tensor };
        const outputNames = detectorSess.outputNames;
        const results = await detectorSess.run(feeds, outputNames);
        const detections = processYoloOutput(results);

        if (detections.length === 0) {
            console.log('No clothing items detected in the image.');
            return { error: 'no_item_detected' };
        }

        // Get the best detection
        const bestDetection = detections.reduce((best, current) =>
            current.confidence > best.confidence ? current : best, detections[0]);
        console.log(`Best detection: ${bestDetection.className} (${bestDetection.confidence.toFixed(2)})`);

        // Segment the image based on the best detection
        const segmentedImageUri = await segmentImageWithMask(imageUri, bestDetection);
        console.log("Segmentation complete. Segmented image URI:", segmentedImageUri);
        console.log("--- Detection & Segmentation Finished ---");

        console.log("--- Starting Clothing Classification ---");
        // Step 2: Classification (Category, Attributes, Color, PCA)
        const classificationTensor = await prepareImageForOnnx(segmentedImageUri);

        const classificationFeeds = { input: classificationTensor };
        const classificationResults = await classificationSess.run(classificationFeeds);

        console.log('===== CLASSIFICATION ONNX RAW OUTPUT =====');
        console.log('Output keys:', Object.keys(classificationResults));
        
        const attributes = classificationResults['attributes'];
        const category = classificationResults['category'];
        const colors = classificationResults['colors'];
        
        console.log('Category tensor shape:', category.dims);
        console.log('Category raw data:', Array.from(category.data as Float32Array));
        
        console.log('Attributes tensor shape:', attributes.dims);
        console.log('Attributes raw data (first 20 values):', Array.from(attributes.data as Float32Array).slice(0, 20));
        console.log('Attributes raw data length:', (attributes.data as Float32Array).length);
        
        console.log('Colors tensor shape:', colors.dims);
        console.log('Colors raw data (first 50 values):', Array.from(colors.data as Float32Array).slice(0, 50));
        console.log('Colors raw data length:', (colors.data as Float32Array).length);
        console.log('=======================================');
        
        // Process category
        const categoryLogits = category.data as Float32Array;
        const categoryProbs = softmax(categoryLogits);
        const categoryIndex = categoryProbs.indexOf(Math.max(...categoryProbs));
        const predictedCategory = category_names[categoryIndex];

        // Process attributes
        const attributeProbs = attributes.data as Float32Array;
        const topAttributes = Array.from(attributeProbs)
            .map((prob, index) => ({ prob, index }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 5)

        // Process colors
        const colorData = colors.data as Float32Array;
        console.log('Fixed processing for colors with shape:', colors.dims);
        
        const colorEntries: Float32Array[] = [];
        const entriesPerColor = 4;
        const numColors = colorData.length / entriesPerColor;
        
        for (let i = 0; i < numColors; i++) {
            colorEntries.push(new Float32Array(colorData.slice(i * entriesPerColor, (i + 1) * entriesPerColor)));
        }
        
        console.log('Created', colorEntries.length, 'color entries');
        const decodedColors = decodeColorResults(colorEntries);

        const calculatedPcaScores = calculatePcaScores(Array.from(attributeProbs));
        console.log('PCA Scores (calculated manually):', calculatedPcaScores.slice(0, 10));
        const principal_components = calculatedPcaScores;
        console.log("Classification Finished");


        console.log("Starting Context Prediction");
        // Step 3: Context Prediction
        let predictedFormality: number | undefined = undefined;
        let predictedTemperature: number | undefined = undefined;
        let predictedSeasons: string[] = [];

        try {
            console.log("Scaling PCA values for context model...");
            const scaledPcaValues = scalePcaValues(
                principal_components,
                contextScalars.mean,
                contextScalars.scale
            );

            console.log('Scaled PCA values:', Array.from(scaledPcaValues));

            // Prepare input tensor for context model
            const contextInputName = contextSess.inputNames[0];
            const contextInputTensor = new ort.Tensor('float32', scaledPcaValues, [1, scaledPcaValues.length]);
            const contextFeeds = { [contextInputName]: contextInputTensor };
            console.log(`Context model input name: ${contextInputName}, Input shape: [1, ${scaledPcaValues.length}]`);

            // Run context predictor inference
            console.log("Running context predictor inference...");
            const contextOutputNames = contextSess.outputNames; // USE passed-in session
            console.log("Context model expected outputs:", contextOutputNames);
            const contextResults = await contextSess.run(contextFeeds, contextOutputNames); // USE passed-in session
            console.log('Context predictor raw outputs:', Object.keys(contextResults));

            // Process context predictor outputs
            const formalityOutput = contextResults['formality'];
            const temperatureOutput = contextResults['temperature'];
            const seasonsOutput = contextResults['seasons'];

            if (formalityOutput) {
                const formalityLogits = formalityOutput.data as Float32Array;
                const formalityProbs = softmax(formalityLogits);
                const formalityIndex = formalityProbs.indexOf(Math.max(...formalityProbs));
                predictedFormality = formalityIndex;
                console.log(`Predicted Formality: ${formality_levels[formalityIndex]} (Index: ${formalityIndex}, Probs: ${Array.from(formalityProbs).map(p=>p.toFixed(2))})`);
                console.log(`FORMALITY RAW INDEX: ${formalityIndex}`);
            } else {
                 console.warn("Context model did not output 'formality'.");
            }

            if (temperatureOutput) {
                const temperatureLogits = temperatureOutput.data as Float32Array;
                const temperatureProbs = softmax(temperatureLogits);
                const temperatureIndex = temperatureProbs.indexOf(Math.max(...temperatureProbs));
                predictedTemperature = 4 - temperatureIndex;
                console.log(`Predicted Temperature: ${temperature_levels[predictedTemperature]} (Model Index: ${temperatureIndex}, Flipped Index: ${predictedTemperature}, Probs: ${Array.from(temperatureProbs).map(p=>p.toFixed(2))})`);
                console.log(`TEMPERATURE RAW INDICES: Model Index=${temperatureIndex}, Flipped Index=${predictedTemperature}`);
            } else {
                 console.warn("Context model did not output 'temperature'.");
            }

            if (seasonsOutput) {
                const seasonProbs = seasonsOutput.data as Float32Array;
                const seasonThreshold = 0.5;
                predictedSeasons = season_names.filter((_, index) => seasonProbs[index] > seasonThreshold);
                console.log(`Predicted Seasons: ${predictedSeasons.join(', ') || 'None'} (Probs: ${Array.from(seasonProbs).map(p=>p.toFixed(2))}, Threshold: ${seasonThreshold})`);
            } else {
                 console.warn("Context model did not output 'seasons'.");
            }

        } catch (contextError) {
            console.error('Context prediction failed:', contextError);
            // Don't throw; allow returning results without context info
        }
        console.log("Context Prediction Finished");


        // === Final Result ===
        const finalPrediction: PredictionResult = {
            category: predictedCategory,
            attributes: topAttributes,
            colors: decodedColors,
            principalComponents: principal_components,
            formality: predictedFormality,
            temperature: predictedTemperature,
            seasons: predictedSeasons,
        };

        if (bestDetection.className === "short_sleeve_top" || categoryIndex === 0) {
            console.log('Special validation for T-shirt (short_sleeve_top) data');
            
            // Ensure formality is a valid number (1-5) or undefined
            if (predictedFormality !== undefined && 
                (!Number.isFinite(predictedFormality) || predictedFormality < 0 || predictedFormality > 4)) {
                finalPrediction.formality = undefined;
                console.log('Fixed invalid formality value for T-shirt');
            }
            
            // Ensure temperature is a valid number (1-5) or undefined
            if (predictedTemperature !== undefined && 
                (!Number.isFinite(predictedTemperature) || predictedTemperature < 0 || predictedTemperature > 4)) {
                finalPrediction.temperature = undefined;
                console.log('Fixed invalid temperature value for T-shirt');
            }
            
            // Ensure seasons is a valid string array or empty array (never undefined)
            finalPrediction.seasons = Array.isArray(predictedSeasons) ? predictedSeasons.filter(s => typeof s === 'string') : [];
        }

        // Define finalProcessedUri only once
        const finalProcessedUri = segmentedImageUri; // Use the segmented image URI

        console.log('--- Classification Complete ---');
        console.log('Final Prediction:', finalPrediction);

        // Return the prediction result
        return { 
          prediction: finalPrediction, 
          processedUri: finalProcessedUri,
          className: bestDetection.className
        };

    } catch (error: any) {
        if (error.message === 'unsupported_image_format') {
            console.log('[classifyImage] Handling unsupported image format');
            // Only skip truly unsupported stuff (just double check)
            if (!imageUri.toLowerCase().endsWith('.png')) {
                return { error: 'unsupported_image_format' };
            }
            console.log('[classifyImage] PNG format detected, attempting to process');
        }
        console.error('[classifyImage] Error during processing:', error);
        return { error: error.message || 'processing_error' };
    }
}

const numColumns = 3;
const screenWidth = Dimensions.get('window').width;
const itemWidth = (screenWidth - 40 - (numColumns - 1) * 10) / numColumns;

async function saveImageToPersistentStorage(uri: string) {
  try {
    const extMatch = uri.match(/\.(jpe?g|png)$/i);
    const ext = extMatch ? extMatch[1].toLowerCase() : 'jpg';
    await RNFS.mkdir(PERSISTENT_IMAGE_DIR);
    const timestamp = Date.now();
    const randomString = Math.random().toString(36).substring(7);
    const filename = `image_${timestamp}_${randomString}.${ext}`;
    const destPath = `${PERSISTENT_IMAGE_DIR}/${filename}`;
    await RNFS.copyFile(uri, destPath);
    return destPath;
  } catch (error) {
    console.error('Error saving image to persistent storage:', error);
    throw error;
  }
}

const calculateSimilarity = (item1: WardrobeItem, item2: WardrobeItem): number => {
  if (!item1.principalComponents || !item2.principalComponents) return 0;
  
  // Calculate Euclidean distance between PCA vectors
  const componentsToUse = Math.min(15, item1.principalComponents.length, item2.principalComponents.length);
  
  let sumSquaredDifferences = 0;
  for (let i = 0; i < componentsToUse; i++) {
    sumSquaredDifferences += Math.pow(item1.principalComponents[i] - item2.principalComponents[i], 2);
  }
  
  const distance = Math.sqrt(sumSquaredDifferences);

  const scale = 50;
  return Math.exp(-distance/scale);
};

// Add function to find most similar item in wardrobe
const findSimilarItem = (targetItem: WardrobeItem, allItems: WardrobeItem[]): WardrobeItem | null => {
  if (!targetItem.principalComponents || allItems.length <= 1) return null;
  
  let mostSimilarItem: WardrobeItem | null = null;
  let highestSimilarity = -1;
  
  allItems.forEach(item => {
    // Skip comparing with self
    if (item.id === targetItem.id) return;
    
    const similarity = calculateSimilarity(targetItem, item);
    if (similarity > highestSimilarity) {
      highestSimilarity = similarity;
      mostSimilarItem = item;
    }
  });
  
  console.log(`Most similar item has similarity score: ${highestSimilarity}`);
  return mostSimilarItem;
};

function generateMask(coeffs: Float32Array, protoData: Float32Array, protoH: number, protoW: number): Float32Array {
  try {
    // The size of the mask
    const maskSize = protoH * protoW;
    
    // Create a new array to hold the mask values
    const mask = new Float32Array(maskSize);
    
    console.log(`Generating mask from ${coeffs.length} coefficients and prototype of shape [1, ${coeffs.length}, ${protoH}, ${protoW}]`);
    console.log('Original coeffs:', Array.from(coeffs));
    
    // For each pixel in the mask
    for (let i = 0; i < maskSize; i++) {
      let sum = 0;
      
      // For each prototype/coefficient
      for (let j = 0; j < coeffs.length; j++) {
        // Skip NaN coefficients
        if (isNaN(coeffs[j])) {
          console.warn(`NaN coefficient at index ${j}, replacing with 0`);
          continue;
        }
        
        // Calculate proper indexing into the flattened prototype tensor
        const protoIdx = j * maskSize + i;
        
        // Skip NaN values in proto data
        if (isNaN(protoData[protoIdx])) {
          console.warn(`Found NaN in proto data at index ${protoIdx}`);
          continue;
        }
        
        sum += coeffs[j] * protoData[protoIdx];
      }
      
      // Apply sigmoid activation to convert to probability
      // Clamp to prevent numerical issues
      const clampedSum = Math.max(-50, Math.min(50, sum));
      mask[i] = 1.0 / (1.0 + Math.exp(-clampedSum));
    }
    
    // Log some stats about the generated mask for debugging
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let numPositive = 0;
    let nanCount = 0;
    
    for (let i = 0; i < maskSize; i++) {
      if (isNaN(mask[i])) {
        nanCount++;
        mask[i] = 0;
        continue;
      }
      
      if (mask[i] < min) min = mask[i];
      if (mask[i] > max) max = mask[i];
      sum += mask[i];
      if (mask[i] > 0.5) numPositive++;
    }
    
    const stats = {
      min,
      max,
      avg: sum / (maskSize - nanCount),
      numPositive,
      nanCount
    };
    
    console.log('Mask generation stats:', stats);
    console.log(`Positive mask pixels: ${stats.numPositive}/${maskSize} (${(stats.numPositive/maskSize*100).toFixed(2)}%)`);
    
    return mask;
  } catch (error) {
    console.error('Error generating mask:', error);
    return new Float32Array(protoH * protoW);
  }
}

function scalePcaValues(
    pcaValues: Float32Array | number[],
    mean: number[],
    scale: number[]
): Float32Array {
    if (pcaValues.length !== mean.length || pcaValues.length !== scale.length) {
        console.warn(`PCA values length (${pcaValues.length}) does not match scaler mean (${mean.length}) or scale (${scale.length}) length. Scaling might be incorrect.`);
    }

    const scaledValues = new Float32Array(pcaValues.length);
    const numComponentsToScale = Math.min(pcaValues.length, mean.length, scale.length);

    for (let i = 0; i < numComponentsToScale; i++) {
        const scaleValue = scale[i];
        if (scaleValue === 0) {
             scaledValues[i] = 0;
             console.warn(`Scaler scale is zero at index ${i}. Setting scaled value to 0.`);
        } else {
            scaledValues[i] = (pcaValues[i] - mean[i]) / scaleValue;
        }
    }

     if (numComponentsToScale < pcaValues.length) {
        for (let i = numComponentsToScale; i < pcaValues.length; i++) {
            scaledValues[i] = pcaValues[i];
        }
     }

    const scaledStats = debugTensorData(scaledValues);
    console.log('Scaled PCA stats:', scaledStats);
    if (scaledStats.hasNaN) {
        console.error("NaN detected in scaled PCA values!");
    }

    return scaledValues;
}

function WardrobeScreen({ navigation }: Props) {
  const [items, setItems] = useState<WardrobeItem[]>([]);
  const [sections, setSections] = useState<CategorySection[]>([]);
  const [selectedItem, setSelectedItem] = useState<WardrobeItem | null>(null);
  const [isImageExpanded, setIsImageExpanded] = useState(false);
  const [similarItem, setSimilarItem] = useState<WardrobeItem | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState('');
  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [selectedItemIds, setSelectedItemIds] = useState<Set<string>>(new Set());

  const [detectorSession, setDetectorSession] = useState<ort.InferenceSession | null>(null);
  const [classifierSession, setClassifierSession] = useState<ort.InferenceSession | null>(null);
  const [contextSession, setContextSession] = useState<ort.InferenceSession | null>(null);
  const [modelsLoaded, setModelsLoaded] = useState(false); // Track loading state

  const [editingCategory, setEditingCategory] = useState(false);

  useEffect(() => {
    const ensurePersistentDirectory = async () => {
      try {
        const exists = await RNFS.exists(PERSISTENT_IMAGE_DIR);
        if (!exists) {
          console.log('Creating persistent directory at:', PERSISTENT_IMAGE_DIR);
          await RNFS.mkdir(PERSISTENT_IMAGE_DIR);
        } else {
          console.log('Persistent directory exists at:', PERSISTENT_IMAGE_DIR);
        }
      } catch (error) {
        console.error('Error ensuring persistent directory:', error);
      }
    };

    // Load ONNX models
    const loadModels = async () => {
      if (modelsLoaded || (!detectorSession && !classifierSession && !contextSession && modelsLoaded)) return; // Simple check

      try {
        console.log('Initializing ONNX models...');
        setModelsLoaded(false);

        const detectorPath = await getModelPath('detector');
        const classifierPath = await getModelPath('classifier');
        const contextPath = await getModelPath('context');

        console.log('Paths:', detectorPath, classifierPath, contextPath);

        const detSess = await ort.InferenceSession.create(detectorPath);
        console.log('Detector session created');
        const clsSess = await ort.InferenceSession.create(classifierPath);
        console.log('Classifier session created');
        const ctxSess = await ort.InferenceSession.create(contextPath);
        console.log('Context session created');

        setDetectorSession(detSess);
        setClassifierSession(clsSess);
        setContextSession(ctxSess);
        setModelsLoaded(true); // Mark as loaded
        console.log('ONNX models initialized successfully.');

      } catch (error) {
        console.error("Failed to initialize ONNX models:", error);
        Alert.alert("Model Error", "Failed to load AI models. Please restart the app.");
        setModelsLoaded(false);
      }
    };

    // Run setup functions
    ensurePersistentDirectory();
    loadSavedItems();
    loadModels();

  }, []);

  const organizeItemsIntoSections = (items: WardrobeItem[]) => {
    // Create a map to group items by category
    const categoryMap = new Map<string, WardrobeItem[]>();
    
    // Initialise with all categories in order
    category_names.forEach(category => {
      categoryMap.set(category, []);
    });

    // Add misc category for unclassified items
    categoryMap.set('Misc', []);

    // Sort items into categories
    items.forEach(item => {
      const category = item.category || 'Misc';
      if (categoryMap.has(category)) {
        categoryMap.get(category)?.push(item);
      } else {
        categoryMap.get('Misc')?.push(item);
      }
    });

    // Convert to sections array with original category order
    return category_names.map(category => ({
      title: category,
      data: categoryMap.get(category) || []
    })).filter(section => section.data.length > 0);
  };

  // Debug function to print all wardrobe data
  const debugPrintWardrobeData = () => {
    console.log('========== WARDROBE DATA DUMP ==========');
    console.log(`Total Items: ${items.length}`);
    
    items.forEach((item, index) => {
      console.log(`\n----- ITEM ${index + 1} (${item.id}) -----`);
      console.log(`Category: ${item.category || 'undefined'}`);
      console.log(`Class Name: ${item.className || 'undefined'}`);
      console.log(`Formality: index ${item.formality !== undefined ? item.formality : 'undefined'} (${item.formality !== undefined ? formality_levels[item.formality] : 'N/A'})`);
      console.log(`Temperature: index ${item.temperature !== undefined ? item.temperature : 'undefined'} (${item.temperature !== undefined ? temperature_levels[item.temperature] : 'N/A'})`);
      
      if (Array.isArray(item.seasons)) {
        console.log(`Seasons: [${item.seasons.join(', ')}]`);
      } else {
        console.log('Seasons: undefined');
      }
      console.log(`Colors: ${item.colors ? JSON.stringify(item.colors.slice(0, 3)) : 'undefined'}`);
    });
    console.log('======================================');
  };

  const loadSavedItems = async () => {
    try {
      const savedItems = await AsyncStorage.getItem('wardrobeItems');
      if (savedItems) {
        const parsedItems = JSON.parse(savedItems);
        setItems(parsedItems);
        setSections(organizeItemsIntoSections(parsedItems));
        
        // Print wardrobe data for debugging
        setTimeout(debugPrintWardrobeData, 1000);
      }
    } catch (error) {
      console.error('Error loading wardrobe items:', error);
      Alert.alert('Error', 'Failed to load wardrobe items');
    }
  };

  const saveItems = async (newItems: WardrobeItem[]) => {
    try {
      const updatedItems = [...newItems]; // Ensure working with a new array reference
      
      // Sanitise each item before saving to prevent rendering issues
      const sanitizedItems = updatedItems.map(item => {
        // Create a clean copy of the item
        const cleanItem: WardrobeItem = {
          id: item.id || `${Date.now()}-${Math.random()}`,
          uri: item.uri || '',
          processedUri: item.processedUri || '',
          category: item.category || undefined,
          className: item.className || undefined,
          colors: Array.isArray(item.colors) ? item.colors.filter(color => 
            Array.isArray(color) && typeof color[0] === 'string' && typeof color[1] === 'number'
          ) : undefined,
          principalComponents: Array.isArray(item.principalComponents) ? item.principalComponents : undefined,
          formality: typeof item.formality === 'number' ? item.formality : undefined,
          temperature: typeof item.temperature === 'number' ? item.temperature : undefined,
          seasons: Array.isArray(item.seasons) ? item.seasons.filter(season => typeof season === 'string') : undefined,
        };
        return cleanItem;
      });
      
      await AsyncStorage.setItem('wardrobeItems', JSON.stringify(sanitizedItems));
      
      if (sanitizedItems.length > 0) {
        const lastItem = sanitizedItems[sanitizedItems.length - 1];
        console.log('Last saved wardrobe item:', JSON.stringify(lastItem, null, 2));
      }
      
      setItems(sanitizedItems);
      setSections(organizeItemsIntoSections(sanitizedItems));
      console.log(`Items saved and sections updated. Total items: ${sanitizedItems.length}`);
      
    } catch (error) {
      console.error('Error saving wardrobe items:', error);
      Alert.alert('Error', 'Failed to save wardrobe items');
    }
  };

  const deleteItemFiles = async (itemToDelete: WardrobeItem) => {
     if (itemToDelete) {
        try {
          if (itemToDelete.uri && typeof itemToDelete.uri === 'string') {
             console.log(`Deleting original file: ${itemToDelete.uri}`);
             await RNFS.unlink(itemToDelete.uri).catch(err => console.log(`Failed to delete original image ${itemToDelete.id}:`, err));
          }
        } catch (e) { console.error("Error deleting original file:", e)}
         try {
           if (itemToDelete.processedUri && typeof itemToDelete.processedUri === 'string') {
             console.log(`Deleting processed file: ${itemToDelete.processedUri}`);
             await RNFS.unlink(itemToDelete.processedUri).catch(err => console.log(`Failed to delete processed image ${itemToDelete.id}:`, err));
           }
         } catch (e) { console.error("Error deleting processed file:", e)}
     }
  }

  const deleteSingleItem = async (itemId: string) => {
    try {
      const itemToDelete = items.find(item => item.id === itemId);
      if (itemToDelete) {
          await deleteItemFiles(itemToDelete);
      }
      const newItems = items.filter((item) => item.id !== itemId);
      await saveItems(newItems);
      setIsImageExpanded(false);
      setSelectedItem(null);
    } catch (error) {
      console.error('Error deleting single item:', error);
      Alert.alert('Error', 'Failed to delete item');
    }
  };

  const toggleItemSelection = (itemId: string) => {
    setSelectedItemIds(prevSelectedIds => {
      const newSelectedIds = new Set(prevSelectedIds);
      if (newSelectedIds.has(itemId)) {
        newSelectedIds.delete(itemId);
      } else {
        newSelectedIds.add(itemId);
      }
      // If last item is deselected, exit selection mode
      if (newSelectedIds.size === 0) {
          setIsSelectionMode(false);
      }
      return newSelectedIds;
    });
  };

  const startSelectionMode = (itemId: string) => {
    setIsSelectionMode(true);
    setSelectedItemIds(new Set([itemId]));
  };

  const handleCancelSelection = () => {
    setIsSelectionMode(false);
    setSelectedItemIds(new Set());
  };

  const handleBatchDelete = async () => {
      const itemsToDeleteCount = selectedItemIds.size;
      if (itemsToDeleteCount === 0) return;

      Alert.alert(
          'Delete Items',
          `Are you sure you want to delete ${itemsToDeleteCount} item(s)?`,
          [
              { text: 'Cancel', style: 'cancel' },
              {
                  text: 'Delete', style: 'destructive',
                  onPress: async () => {
                      const itemsToKeep = items.filter(item => !selectedItemIds.has(item.id));
                      const itemsToDelete = items.filter(item => selectedItemIds.has(item.id));

                      // Attempt to delete files first
                      for (const item of itemsToDelete) {
                          await deleteItemFiles(item);
                      }

                      // Update state and save
                      await saveItems(itemsToKeep);

                      // Exit selection mode
                      handleCancelSelection();
                      console.log(`Batch deleted ${itemsToDeleteCount} items.`);
                  }
              },
          ]
      );
  };


  // --- Item Press Handling ---
  // Decides whether to open modal or toggle selection based on mode
  const handleItemPress = (item: WardrobeItem) => {
    if (isSelectionMode) {
      toggleItemSelection(item.id);
    } else {
      // Original behavior: open modal and find similar
      setSelectedItem(item);
      setIsImageExpanded(true);
      const similar = findSimilarItem(item, items);
      setSimilarItem(similar);
      
      // Log all values of the expanded item
      console.log('====== EXPANDED ITEM VALUES ======');
      console.log(`ID: ${item.id}`);
      console.log(`Original URI: ${item.uri}`);
      console.log(`Processed URI: ${item.processedUri}`);
      console.log(`Category: ${item.category || 'undefined'}`);
      console.log(`Class Name: ${item.className || 'undefined'}`);
      console.log(`Formality: index ${item.formality !== undefined ? item.formality : 'undefined'} (${item.formality !== undefined ? formality_levels[item.formality] : 'N/A'})`);
      console.log(`Temperature: index ${item.temperature !== undefined ? item.temperature : 'undefined'} (${item.temperature !== undefined ? temperature_levels[item.temperature] : 'N/A'})`);
      
      if (Array.isArray(item.seasons)) {
        console.log(`Seasons: [${item.seasons.join(', ')}]`);
      } else {
        console.log('Seasons: undefined');
      }
      
      if (item.colors) {
        console.log('Colors:');
        item.colors.forEach((color, index) => {
          console.log(`  ${index+1}. ${color[0]} (Percentage: ${(color[1] * 100).toFixed(2)}%, RGB: [${color[2].join(', ')}])`);
        });
      } else {
        console.log('Colors: undefined');
      }
      
      if (item.principalComponents) {
        console.log('Principal Components:');
        console.log(item.principalComponents);
      } else {
        console.log('Principal Components: undefined');
      }
      console.log('================================');
    }
  };

  const renderItem = ({ item }: { item: WardrobeItem }) => {
    const isSelected = selectedItemIds.has(item.id);

    return (
      <TouchableOpacity
        style={[styles.gridItem, isSelected && styles.selectedGridItem]} // Apply selection style
        onPress={() => handleItemPress(item)}
        onLongPress={() => startSelectionMode(item.id)} // Activate selection mode on long press
        delayLongPress={300} // Standard long press delay
      >
        <Image source={{ uri: item.processedUri }} style={styles.image} />
        {item.category && !isSelectionMode && ( // Hide category label in selection mode for clarity
          <View style={styles.categoryLabel}>
            <Text style={styles.categoryText}>{item.category}</Text>
          </View>
        )}
        {isSelected && ( // Show checkmark overlay if selected
             <View style={styles.selectionOverlay}>
                 <Text style={styles.checkmark}>✓</Text>
             </View>
         )}
      </TouchableOpacity>
    );
  };

  const takePhoto = async () => {
    // Check if models are loaded before proceeding
    if (!modelsLoaded || !detectorSession || !classifierSession || !contextSession) {
      Alert.alert("Models Not Ready", "The AI models are still loading. Please try again in a moment.");
      return;
    }
    setIsProcessing(true); // Start processing indicator
    try {
      const result = await ImagePicker.launchCamera({
        mediaType: 'photo',
        quality: 1.0,
        maxWidth: 1024,
        maxHeight: 1024,
        saveToPhotos: false,
        includeBase64: false,
        includeExtra: true,
      });

      if (result.didCancel || !result.assets?.[0]?.uri) {
        setIsProcessing(false); // Stop processing if cancelled
        return;
      }

      const originalUri = await saveImageToPersistentStorage(result.assets[0].uri);
      // Pass loaded sessions to classifyImage
      const classificationResult = await classifyImage(
          originalUri,
          detectorSession,
          classifierSession,
          contextSession
      );

      // Check if there was an error during classification
      if ('error' in classificationResult) {
        console.log(`Classification issue: ${classificationResult.error}`);
        // Handle specific error types
        if (classificationResult.error === 'no_item_detected') {
          setProcessingStatus('No clothing items found in image.');
          // Optionally add a small delay to show the status before hiding the processing indicator
          setTimeout(() => {
            setProcessingStatus('');
            setIsProcessing(false);
          }, 1500);
          return;
        } else if (classificationResult.error === 'unsupported_image_format') {
          // For non-PNG unsupported formats
          setProcessingStatus('This image format is not supported. Please use photos from your camera or JPEG/PNG images.');
          setTimeout(() => {
            setProcessingStatus('');
            setIsProcessing(false);
          }, 2500);
          return;
        }
        // For other errors, continue to the catch block
        throw new Error(classificationResult.error);
      }

      const { prediction, processedUri, className } = classificationResult;

      const newItem: WardrobeItem = {
        id: Date.now().toString(),
        uri: originalUri,
        processedUri: processedUri,
        category: prediction.category,
        className: className,
        colors: prediction.colors,
        principalComponents: prediction.principalComponents,
        formality: prediction.formality,
        temperature: prediction.temperature,
        seasons: prediction.seasons
      };

      const newItems = [...items, newItem];
      setItems(newItems);
      await saveItems(newItems);

    } catch (error) {
      console.error('[takePhoto] Failed to process photo:', error);
      setProcessingStatus('Could not process image.');
      setTimeout(() => {
        setProcessingStatus('');
      }, 1500);
    } finally {
      setIsProcessing(false);
    }
  };

  const selectPhoto = async () => {
    // Check if models are loaded before proceeding
    if (!modelsLoaded || !detectorSession || !classifierSession || !contextSession) {
      Alert.alert("Models Not Ready", "The AI models are still loading. Please try again in a moment.");
      return;
    }

    setIsProcessing(true);
    setProcessingStatus('Starting...');
    let finalItems: WardrobeItem[] = [];

    try {
      const result = await ImagePicker.launchImageLibrary({
        mediaType: 'photo',
        quality: 1.0,
        maxWidth: 1024,
        maxHeight: 1024,
        selectionLimit: 0,
      });

      // Check if cancelled or no assets selected
      if (result.didCancel || !result.assets || result.assets.length === 0) {
        setIsProcessing(false);
        setProcessingStatus('');
        return;
      }

      const totalImages = result.assets.length;
      let processedCount = 0;
      let skippedCount = 0;

      finalItems = [...items];

      for (let i = 0; i < totalImages; i++) {
        const asset = result.assets[i];
        const statusText = totalImages === 1
          ? 'Processing image...'
          : `Processing image ${i + 1} of ${totalImages}...`;
        setProcessingStatus(statusText);

        if (asset.uri) {
          try {
            console.log(`Processing image: ${asset.fileName || asset.uri}`);
            const originalUri = await saveImageToPersistentStorage(asset.uri);

            // Pass loaded sessions to classifyImage
            const classificationResult = await classifyImage(
                originalUri,
                detectorSession,
                classifierSession,
                contextSession
            );

            // Check if there was an error during classification
            if ('error' in classificationResult) {
              console.log(`Classification issue for image ${i+1}: ${classificationResult.error}`);
              
              if (classificationResult.error === 'no_item_detected') {
                // Graceful handling for no item detected
                if (totalImages === 1) {
                  setProcessingStatus('No clothing items found in image.');
                } else {
                  setProcessingStatus(`No clothing items found in image ${i+1}. Skipping...`);
                  // Wait a moment so user can see the message
                  await new Promise(resolve => setTimeout(resolve, 1000));
                }
                skippedCount++;
                continue; // Skip to next image
              } else if (classificationResult.error === 'unsupported_image_format') {
                // Graceful handling for unsupported image format
                if (totalImages === 1) {
                  setProcessingStatus('This image format is not supported. Please use photos from your camera or JPEG/PNG images.');
                } else {
                  setProcessingStatus(`Image ${i+1} has an unsupported format. Skipping...`);
                  // Wait a moment so user can see the message
                  await new Promise(resolve => setTimeout(resolve, 1000));
                }
                skippedCount++;
                continue; // Skip to next image
              }
              
              // For other errors, throw to be caught by the catch block
              throw new Error(classificationResult.error);
            }

            const { prediction, processedUri, className } = classificationResult;

            const newItem: WardrobeItem = {
              id: `${Date.now()}-${Math.random()}`,
              uri: originalUri,
              processedUri: processedUri,
              category: prediction.category,
              className: className,
              colors: prediction.colors,
              principalComponents: prediction.principalComponents,
              formality: prediction.formality,
              temperature: prediction.temperature,
              seasons: prediction.seasons
            };

            finalItems.push(newItem);
            await saveItems(finalItems);

            // Add a short delay to allow potential garbage collection/resource release
            await new Promise(resolve => setTimeout(resolve, 100)); // Keep the delay for now

            processedCount++;

          } catch (singleImageError) {
             // Log error but don't show intrusive alerts
             console.error(`Failed to process image ${asset.fileName || asset.uri}:`, singleImageError);
             
             // Show a status message instead of an alert
             if (totalImages === 1) {
               setProcessingStatus('Could not process image.');
             } else {
               setProcessingStatus(`Could not process image ${i+1}. Skipping...`);
               // Wait a moment so user can see the message
               await new Promise(resolve => setTimeout(resolve, 1000));
             }
             skippedCount++;
          }
        }
      }

      // Log completion message based on processed count
      if (processedCount > 0) {
          const summary = skippedCount > 0 
            ? `Added ${processedCount} items (${skippedCount} skipped)`
            : `Successfully added ${processedCount} items`;
          console.log(summary);
          setProcessingStatus(summary);
          // Show summary briefly before clearing
          setTimeout(() => setProcessingStatus(''), 1500);
      } else {
          console.log('No items were successfully processed.');
          if (skippedCount > 0) {
            setProcessingStatus('No valid clothing items found in images');
            setTimeout(() => setProcessingStatus(''), 1500);
          }
      }

    } catch (error) {
      console.error('[selectPhoto] Failed to process photos:', error);
      setProcessingStatus('Error processing photos');
      setTimeout(() => setProcessingStatus(''), 1500);
    } finally {
      setTimeout(() => {
        setIsProcessing(false);
        setProcessingStatus('');
      }, 1600);
    }
  };

  const showImageSourceOptions = () => {
    Alert.alert(
      'Add Photo',
      'Choose photo source',
      [
        {
          text: 'Camera',
          onPress: () => {
            console.log('Starting camera...');
            requestAnimationFrame(takePhoto);
          },
        },
        {
          text: 'Photo Library',
          onPress: selectPhoto,
        },
        {
          text: 'Cancel',
          style: 'cancel',
        },
      ],
    );
  };

  // Update the expanded image modal to use deleteSingleItem
  const renderExpandedImage = () => (
    <Modal
      visible={isImageExpanded && !isSelectionMode}
      transparent={true}
      animationType='fade'
      onRequestClose={() => {
        setIsImageExpanded(false);
        setEditingCategory(false);
      }}
    >
      <View style={styles.modalOverlay}>
        {/* Background touchable that only covers the empty space */}
        <TouchableWithoutFeedback onPress={() => {
          setIsImageExpanded(false);
          setEditingCategory(false);
        }}>
          <View style={StyleSheet.absoluteFill} />
        </TouchableWithoutFeedback>

        {/* Modal content container - NOT wrapped in Touchable */}
        <View style={styles.expandedImageContainer}>
          {selectedItem && (
            <>
              <ScrollView 
                style={styles.scrollContainer}
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={true}
                bounces={true}
                overScrollMode='always'
                nestedScrollEnabled={true}
              >
                <View style={styles.contentWrapper}>
                  <Image
                    source={{ uri: selectedItem.processedUri }}
                    style={styles.expandedImage}
                    resizeMode='contain'
                  />
                  
                  {/* Category Section */}
                  <View style={styles.categoryContainer}>
                    <View style={styles.categoryHeader}>
                      <Text style={styles.categoryTitle}>
                        {selectedItem.category || 'Uncategorized'}
                      </Text>
                      {!editingCategory && (
                        <TouchableOpacity 
                          style={styles.editCategoryButton}
                          onPress={() => setEditingCategory(true)}
                        >
                          <Text style={styles.editCategoryText}>Edit</Text>
                        </TouchableOpacity>
                      )}
                    </View>
                    
                    {editingCategory && (
                      <View style={styles.categorySelector}>
                        <ScrollView 
                          horizontal={true}
                          showsHorizontalScrollIndicator={false}
                          contentContainerStyle={styles.categorySelectorContent}
                        >
                          {category_names.map((category) => (
                            <TouchableOpacity
                              key={category}
                              style={[
                                styles.categoryOption,
                                selectedItem.category === category && styles.selectedCategoryOption
                              ]}
                              onPress={() => updateItemCategory(selectedItem.id, category)}
                            >
                              <Text 
                                style={[
                                  styles.categoryOptionText,
                                  selectedItem.category === category && styles.selectedCategoryOptionText
                                ]}
                              >
                                {category}
                              </Text>
                            </TouchableOpacity>
                          ))}
                        </ScrollView>
                        <TouchableOpacity 
                          style={styles.cancelEditButton}
                          onPress={() => setEditingCategory(false)}
                        >
                          <Text style={styles.cancelEditText}>Cancel</Text>
                        </TouchableOpacity>
                      </View>
                    )}
                  </View>
                  
                  {/* --- Usage Context Section --- */}
                  <View style={styles.contextContainer}>
                    <Text style={styles.contextTitle}>Usage Context</Text>

                    {/* Formality Scale */}
                    {selectedItem.formality !== undefined && 
                     selectedItem.formality !== null && 
                     Number.isFinite(selectedItem.formality) && (
                      <View style={styles.scaleContainer}>
                        <Text style={styles.scaleLabel}>Formality:</Text>
                        <View style={styles.scaleBarContainer}>
                          <View style={styles.scaleBar}>
                            {FORMALITY_SCALE.map((level, index) => {
                              const isSelected = index === selectedItem.formality;
                              const positionPercent = (index / (FORMALITY_SCALE.length - 1)) * 100;
                              return (
                                <React.Fragment key={level}>
                                  {isSelected && (
                                    <View style={[styles.scaleIndicator, { left: `${positionPercent}%` }]} />
                                  )}
                                </React.Fragment>
                              );
                            })}
                          </View>
                          <View style={styles.scaleLabelsContainer}>
                             {FORMALITY_SCALE.map((level, index) => (
                               <Text key={level} style={[styles.scaleTickLabel, selectedItem.formality === index && styles.scaleTickLabelSelected]}>{level.replace(' ', '\n')}</Text>
                             ))}
                          </View>
                        </View>
                      </View>
                    )}

                    {/* Temperature Scale */}
                    {selectedItem.temperature !== undefined && 
                     selectedItem.temperature !== null && 
                     Number.isFinite(selectedItem.temperature) && (
                      <View style={styles.scaleContainer}>
                        <Text style={styles.scaleLabel}>Temperature Suitability:</Text>
                        <View style={styles.scaleBarContainer}>
                           <LinearGradient
                             colors={['#6495ED', '#90EE90', '#FFD700', '#FFA500', '#FF4500']} // Blue -> Green -> Yellow -> Orange -> Red (Very Cold to Hot)
                             start={{x: 0, y: 0.5}}
                             end={{x: 1, y: 0.5}}
                             style={styles.tempScaleBarGradient}
                           >
                               {TEMPERATURE_SCALE.map((level, index) => {
                                   const isSelected = index === selectedItem.temperature;
                                   // Position calculation to show Very Cold (index 0) on left and Hot (index 4) on right
                                   const positionPercent = (index / (TEMPERATURE_SCALE.length - 1)) * 100;
                                   return (
                                     <React.Fragment key={level}>
                                       {isSelected && (
                                          <View style={[styles.scaleIndicator, styles.tempScaleIndicator, { left: `${positionPercent}%` }]} />
                                       )}
                                     </React.Fragment>
                                   );
                               })}
                           </LinearGradient>
                           <View style={[styles.scaleLabelsContainer]}>
                              {/* Display labels in order (Very Cold to Hot) */}
                              {TEMPERATURE_SCALE.map((level, index) => {
                                return <Text key={level} style={[styles.scaleTickLabel, selectedItem.temperature === index && styles.scaleTickLabelSelected]}>{level.replace(' ', '\n')}</Text>;
                              })}
                            </View>
                        </View>
                      </View>
                    )}

                    {/* Seasons Icons */}
                    {/* Ensure seasons is a valid array before rendering */}
                    {Array.isArray(selectedItem.seasons) && 
                     selectedItem.seasons.length > 0 && 
                     selectedItem.seasons.every(season => typeof season === 'string') && (
                      <View style={styles.seasonsContainer}>
                          <Text style={styles.scaleLabel}>Seasons:</Text>
                          <View style={styles.seasonsIconsContainer}>
                             {selectedItem.seasons.map(season => (
                               <View key={season} style={styles.seasonIconWrapper}>
                                  <Text style={styles.seasonIcon}>{SEASON_ICONS[season] || '?'}</Text>
                                  <Text style={styles.seasonLabel}>{season}</Text>
                               </View>
                             ))}
                          </View>
                       </View>
                    )}

                    {/* Placeholder if no context */}
                    {(!selectedItem.formality || !Number.isFinite(selectedItem.formality)) && 
                     (!selectedItem.temperature || !Number.isFinite(selectedItem.temperature)) && 
                     (!Array.isArray(selectedItem.seasons) || selectedItem.seasons.length === 0) && (
                        <Text style={styles.contextPlaceholder}>No context information available.</Text>
                     )}
                  </View>
                  {/* --- End Usage Context Section --- */}

                  {/* Add similar item section - ensure similarItem is valid */}
                  {similarItem && 
                   typeof similarItem === 'object' && 
                   similarItem.uri && (
                    <View style={styles.similarItemContainer}>
                      <Text style={styles.similarItemTitle}>Similar Item:</Text>
                      <View style={styles.similarItemContent}>
                        <Image 
                          source={{ uri: similarItem.processedUri }} 
                          style={styles.similarItemImage} 
                          resizeMode="cover"
                        />
                        <View style={styles.similarItemInfo}>
                          <Text style={styles.similarItemCategory}>{similarItem.category}</Text>
                        </View>
                      </View>
                    </View>
                  )}

                  {/* Colors section - ensure colors array is valid */}
                  {Array.isArray(selectedItem.colors) && 
                   selectedItem.colors.length > 0 && 
                   selectedItem.colors.every(color => 
                     Array.isArray(color) && 
                     typeof color[0] === 'string' && 
                     typeof color[1] === 'number'
                   ) && (
                    <View style={styles.colorsContainer}>
                      <Text style={styles.colorsTitle}>Dominant Colors:</Text>
                      <View style={styles.colorsList}>
                        {selectedItem.colors.map(([color, percentage, _], index) => (
                          <View key={index} style={styles.colorPill}>
                            <View 
                              style={[
                                styles.colorSwatch,
                                { backgroundColor: COLOR_MAP[color.toLowerCase()] || '#ffffff' }
                              ]} 
                            />
                            <Text style={styles.colorText}>
                              {formatColorName(color)} ({Math.round(percentage * 100)}%)
                            </Text>
                          </View>
                        ))}
                      </View>
                    </View>
                  )}
                </View>
              </ScrollView>

              <View style={styles.imageActions}>
                <TouchableOpacity
                  style={styles.deleteButton}
                  onPress={() => {
                    // Use the single item delete function here
                    Alert.alert('Delete Item', 'Are you sure?', [
                      { text: 'Cancel', style: 'cancel' },
                      { text: 'Delete', style: 'destructive', onPress: () => deleteSingleItem(selectedItem.id) },
                    ]);
                  }}
                >
                  <Text style={styles.deleteButtonText}>DELETE</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.closeExpandedButton}
                  onPress={() => {
                    setIsImageExpanded(false);
                    setEditingCategory(false);
                  }}
                >
                  <Text style={styles.closeButtonText}>CLOSE</Text>
                </TouchableOpacity>
              </View>
            </>
          )}
        </View>
      </View>
    </Modal>
  );

  const updateItemCategory = async (itemId: string, newCategory: string) => {
    try {
      const updatedItems = items.map(item => {
        if (item.id === itemId) {
          return { ...item, category: newCategory };
        }
        return item;
      });
      
      await saveItems(updatedItems);
      setEditingCategory(false);
      
      if (selectedItem && selectedItem.id === itemId) {
        setSelectedItem({...selectedItem, category: newCategory});
      }
    } catch (error) {
      console.error('Error updating item category:', error);
      Alert.alert('Error', 'Failed to update item category');
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
         {!isSelectionMode ? (
           <View style={styles.header}>
             <Text style={styles.title}>My Wardrobe</Text>
             <TouchableOpacity 
               style={[styles.outfitButton, {marginRight: 10}]}
               onPress={() => navigation.navigate('OutfitGenerator')}
             >
               <Text style={styles.outfitButtonText}>Create Outfit</Text>
             </TouchableOpacity>
           </View>
         ) : (
           <View style={styles.selectionHeader}>
             <TouchableOpacity onPress={handleCancelSelection}>
               <Text style={styles.selectionButtonText}>Cancel</Text>
             </TouchableOpacity>
             <Text style={styles.selectionTitle}>{selectedItemIds.size} Selected</Text>
             <TouchableOpacity onPress={handleBatchDelete} disabled={selectedItemIds.size === 0}>
                <Text style={[styles.selectionButtonText, styles.deleteActionText, selectedItemIds.size === 0 && styles.disabledText]}>Delete</Text>
             </TouchableOpacity>
           </View>
        )}

        {/* Main Content List */}
        <FlatList
          data={sections}
          renderItem={({item}) => (
            <>
              <View style={styles.sectionHeader}>
                <Text style={styles.sectionHeaderText}>{item.title}</Text>
              </View>
              <FlatList
                data={item.data}
                renderItem={renderItem}
                keyExtractor={(i) => i.id}
                numColumns={numColumns}
                columnWrapperStyle={styles.row}
                scrollEnabled={!isSelectionMode}
                extraData={selectedItemIds}
              />
            </>
          )}
          keyExtractor={(section) => section.title}
        />

        {/* Hide Add button in selection mode */}
        {!isSelectionMode && (
          <TouchableOpacity
            style={[styles.addButton, !modelsLoaded && styles.disabledButton]}
            onPress={showImageSourceOptions}
            disabled={!modelsLoaded}
           >
            <Text style={styles.addButtonText}>+</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Modals and Overlays */}
      {renderExpandedImage()}
      {isProcessing && (
        <View style={styles.processingOverlay}>
          <ActivityIndicator size="large" color="#000000" />
          <Text style={styles.processingText}>{processingStatus || 'Processing image...'}</Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  content: {
    flex: 1,
    padding: 20,
    paddingBottom: 0,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
    paddingHorizontal: 0,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  outfitButton: {
    backgroundColor: '#000',
    paddingVertical: Math.max(6, screenWidth * 0.015),
    paddingHorizontal: Math.max(12, screenWidth * 0.03),
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  outfitButtonText: {
    color: '#fff',
    fontSize: Math.max(12, screenWidth * 0.035),
    fontWeight: '600',
  },
  selectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
    paddingHorizontal: 5,
    height: 40,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  selectionTitle: {
      fontSize: 16,
      fontWeight: '600',
      color: '#333',
  },
  selectionButtonText: {
      fontSize: 16,
      color: '#007AFF',
      fontWeight: '500',
      padding: 5,
  },
  deleteActionText: {
      color: '#FF3B30',
  },
  disabledText: {
      color: '#aaa',
  },
  grid: {
    paddingBottom: 80,
  },
  row: {
    justifyContent: 'flex-start',
    gap: 10,
    marginBottom: 10,
  },
  gridItem: {
    width: itemWidth,
    height: itemWidth,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  selectedGridItem: {
    borderColor: '#007AFF',
    opacity: 0.8,
  },
  selectionOverlay: {
     ...StyleSheet.absoluteFillObject,
     backgroundColor: 'rgba(0, 122, 255, 0.3)',
     justifyContent: 'center',
     alignItems: 'center',
  },
  checkmark: {
      fontSize: 40,
      color: 'white',
      fontWeight: 'bold',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  categoryLabel: {
    position: 'absolute',
    bottom: 8,
    left: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  categoryText: {
    color: 'white',
    fontSize: 10,
    fontWeight: '500',
    letterSpacing: 0.2,
  },
  addButton: {
    position: 'absolute',
    bottom: 20,
    right: 20,
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  addButtonText: {
    color: '#fff',
    fontSize: 32,
    marginTop: -2,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  expandedImageContainer: {
    width: '90%',
    maxHeight: '85%',
    backgroundColor: '#f5f5f5',
    borderRadius: 20,
    overflow: 'hidden',
  },
  scrollContainer: {
    flexShrink: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingBottom: 24,
  },
  contentWrapper: {
    paddingHorizontal: 16,
  },
  expandedImage: {
    width: '100%',
    height: undefined,
    aspectRatio: 1,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.1)',
    marginBottom: 16,
    marginTop: 16,
  },
  imageActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
    width: '100%',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    backgroundColor: '#f5f5f5',
  },
  deleteButton: {
    backgroundColor: '#FFF0F0',
    borderWidth: 1,
    borderColor: '#FF3B30',
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: 10,
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
  },
  closeExpandedButton: {
    backgroundColor: '#F0F7FF',
    borderWidth: 1,
    borderColor: '#007AFF',
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: 10,
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
  },
  deleteButtonText: {
    color: '#FF3B30',
    fontSize: 15,
    fontWeight: '600',
    letterSpacing: 0.25,
  },
  closeButtonText: {
    color: '#007AFF',
    fontSize: 15,
    fontWeight: '600',
    letterSpacing: 0.25,
  },
  sectionHeader: {
    backgroundColor: '#f8f8f8',
    paddingVertical: 8,
    paddingHorizontal: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
    marginBottom: 10,
  },
  sectionHeaderText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  categoryContainer: {
    paddingVertical: 12,
    backgroundColor: '#f5f5f5',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
    zIndex: 1,
    marginTop: 16,
  },
  categoryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
  },
  categoryTitle: {
    color: '#333',
    fontSize: 18,
    fontWeight: '600',
    letterSpacing: 0.25,
  },
  editCategoryButton: {
    backgroundColor: '#f0f7ff',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#007AFF',
  },
  editCategoryText: {
    color: '#007AFF',
    fontSize: 14,
    fontWeight: '500',
  },
  categorySelector: {
    marginTop: 12,
    paddingBottom: 8,
  },
  categorySelectorContent: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    gap: 8,
  },
  categoryOption: {
    backgroundColor: '#f0f0f0',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 16,
    marginRight: 8,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  selectedCategoryOption: {
    backgroundColor: '#007AFF',
    borderColor: '#0056b3',
  },
  categoryOptionText: {
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
  },
  selectedCategoryOptionText: {
    color: '#fff',
  },
  cancelEditButton: {
    alignSelf: 'center',
    paddingVertical: 12,
    paddingHorizontal: 24,
    marginTop: 16,
    marginHorizontal: 16,
    backgroundColor: '#FFF0F0',
    borderWidth: 1,
    borderColor: '#FF3B30',
    borderRadius: 10,
  },
  cancelEditText: {
    color: '#FF3B30',
    fontSize: 15,
    fontWeight: '600',
    textAlign: 'center',
  },
  attributesContainer: {
    paddingHorizontal: 16,
    marginTop: 16,
    marginBottom: 16,
  },
  attributesTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#444',
    marginBottom: 8,
  },
  attributesList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  attributePill: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: 12,
    paddingVertical: 4,
    paddingHorizontal: 8,
  },
  attributeText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '500',
    letterSpacing: 0.2,
  },
  colorsContainer: {
    paddingHorizontal: 16,
    marginTop: 16,
    marginBottom: 24,
  },
  colorsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#444',
    marginBottom: 12,
  },
  colorsList: {
    flexDirection: 'column',
    gap: 8,
  },
  colorPill: {
    backgroundColor: '#f0f0f0',
    borderRadius: 12,
    padding: 8,
    flexDirection: 'row',
    alignItems: 'center',
    width: '100%',
  },
  colorSwatch: {
    width: 24,
    height: 24,
    borderRadius: 6,
    marginRight: 12,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.1)',
  },
  colorText: {
    color: '#333',
    fontSize: 14,
    fontWeight: '500',
    flex: 1,
  },
  similarItemContainer: {
    marginTop: 20,
    marginBottom: 16,
    padding: 16,
    backgroundColor: '#f9f9f9',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    marginHorizontal: 16,
  },
  similarItemTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  similarItemContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  similarItemImage: {
    width: 80,
    height: 80,
    borderRadius: 8,
    marginRight: 16,
  },
  similarItemInfo: {
    flex: 1,
  },
  similarItemCategory: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  similarItemFeatures: {
    fontSize: 12,
    color: '#666',
    lineHeight: 18,
  },
  processingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 10,
  },
  processingText: {
    marginTop: 10,
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  contextContainer: {
    paddingHorizontal: 16,
    marginTop: 20,
    marginBottom: 16,
    backgroundColor: '#ffffff',
    borderRadius: 10,
    padding: 14,
    borderWidth: 1,
    borderColor: '#eee',
  },
  contextTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
    paddingBottom: 8,
  },
  contextRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  contextLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: '#555',
  },
  contextValue: {
    fontSize: 14,
    color: '#333',
    textAlign: 'right',
    flexShrink: 1,
  },
  contextPlaceholder: {
     fontSize: 14,
     color: '#888',
     fontStyle: 'italic',
     textAlign: 'center',
     marginTop: 5,
  },
  scaleContainer: {
    marginBottom: 20,
  },
  scaleLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: '#555',
    marginBottom: 12,
  },
  scaleBarContainer: {
  },
  scaleBar: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    position: 'relative',
    marginBottom: 8,
  },
  tempScaleBarGradient: {
    height: 8,
    borderRadius: 4,
    position: 'relative',
    marginBottom: 8,
  },
  scaleIndicator: {
    position: 'absolute',
    bottom: -4,
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: '#007AFF',
    borderWidth: 2,
    borderColor: '#fff',
    transform: [{ translateX: -8 }],
    zIndex: 1,
  },
  tempScaleIndicator: {
      backgroundColor: '#444',
  },
  scaleTick: {
    position: 'absolute',
    bottom: -2,
    width: 1,
    height: 4,
    backgroundColor: '#aaa',
    transform: [{ translateX: -0.5 }],
  },
  scaleLabelsContainer: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      paddingHorizontal: 0,
      marginTop: 5,
  },
  scaleTickLabel: {
      fontSize: 10,
      color: '#888',
      textAlign: 'center',
      width: (Dimensions.get('window').width * 0.9 - 60) / 5,
  },
  scaleTickLabelSelected: {
      fontWeight: 'bold',
      color: '#333',
  },
  seasonsContainer: {
    marginTop: 10,
  },
  seasonsIconsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    marginTop: 8,
    paddingVertical: 10,
    backgroundColor: '#f8f8f8',
    borderRadius: 8,
  },
  seasonIconWrapper: {
    alignItems: 'center',
  },
  seasonIcon: {
    fontSize: 24,
  },
  seasonLabel: {
      fontSize: 11,
      color: '#555',
      marginTop: 2,
  },
  disabledButton: {
    backgroundColor: '#cccccc',
    opacity: 0.6,
  },
});

export default WardrobeScreen;
