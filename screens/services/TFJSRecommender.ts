/**
 * This is the TensorFlow.js outfit recommendation engine:
 *
 * - Registers a custom L2Regularizer for model weights penalty.
 * - Loads a bundled model from Expo assets (via Base64 <-> ArrayBuffer) and falls back safely on errors.
 * - Supports asynchronously swapping in a previously trained model from AsyncStorage.
 * - Prepares input tensors (features, masks, colors, and target temperatures) from raw outfit data.
 * - Provides predict(), printScores(), train(), saveModel(), deleteSavedModel(), and resetTensorMemory() methods
 *   with error handling, detailed logging, and safe tensor cleanup to avoid memory leaks.
*/

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Buffer } from 'buffer';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';

const MAX_ITEMS = 3;
const NEW_FEATURE_DIM = 78;
const COLOR_FEATURES_PER_ITEM = 15;

interface ColorInfo {
  rgb?: [number, number, number];
  hsv?: [number, number, number];
  percentage?: number | string;
}

interface OutfitSlot {
  item_present?: number | string;
  pca_values?: { [key: string]: number | string };
  temperature_suitability?: (number | string)[];
  temperature?: number | string;
  colors?: ColorInfo[];
}

type OutfitTensor = OutfitSlot[];

const MODEL_STORAGE_KEY = 'tensorflowjs_models/recommendations_model';

const MODEL_ASSETS = {
    modelJson: require('../assets/model.json'),
    modelWeightsModule: require('../assets/group1-shard1of1.bin')
};

try {
    class L2Regularizer extends tf.serialization.Serializable {
      static className = 'l2';
      private l2Value: number;
  
      constructor(config: {l2: number}) {
        super();
        this.l2Value = config.l2;
      }
  
      apply(weights: tf.Tensor | null | undefined): tf.Tensor {
        try {
          if (!weights) {
            console.error('L2Regularizer received null weights');
            return tf.scalar(0);
          }
          
          return tf.tidy(() => {
            if (weights.isDisposed) {
              console.error('L2Regularizer received disposed weights tensor');
              return tf.scalar(0);
            }
            
            const squared = tf.square(weights);
            const summed = tf.sum(squared);
            return tf.mul(this.l2Value, summed);
          });
        } catch (error) {
          console.error('Error in L2Regularizer.apply:', error);
          return tf.scalar(0);
        }
      }
  
      getConfig(): {l2: number} {
        return {l2: this.l2Value};
      }
  
      static override fromConfig<T extends tf.serialization.Serializable>(
        cls: tf.serialization.SerializableConstructor<T>,
        config: tf.serialization.ConfigDict
      ): T {
        return new L2Regularizer({l2: config['l2'] as number}) as unknown as T;
      }
    }
  
    const classNameMap = (tf.serialization as any).classNameMap || {};
    if (!classNameMap['L2']) {
      tf.serialization.registerClass(L2Regularizer);
      
      console.log('Successfully registered custom L2 regularizer');
    } else {
      console.log('L2 regularizer already registered');
    }
  } catch (error) {
    console.error('Error setting up L2 regularizer:', error);
  }

export class TFJSRecommender {
  protected model: tf.LayersModel | null = null;
  private modelLoaded = false;
  private isLoading = false;
  private isCompiledForTraining = false;
  private isUsingTrainedModel = false;
  private initialLoadComplete = false;
  private modelSwapComplete = false;
  private modelLoadListeners: Array<() => void> = [];

  async waitForCompleteModelLoad(timeout = 10000): Promise<boolean> {
    if (this.modelLoaded && this.modelSwapComplete) {
      console.log("Model already fully loaded and ready for use");
      return true;
    }
    
    // If the model isn't even started loading yet, start the load
    if (!this.isLoading && !this.modelLoaded) {
      console.log("Starting model load from waitForCompleteModelLoad");
      this.loadModel().catch(err => console.error("Failed to load model:", err));
    }
    
    return new Promise<boolean>((resolve) => {
      // Set a timeout in case loading takes too long
      const timeoutId = setTimeout(() => {
        console.log(`⚠️ Model load timeout (${timeout}ms) - proceeding with whatever model is available`);
        resolve(this.modelLoaded);
      }, timeout);
      
      const checkComplete = () => {
        if (this.modelLoaded && this.modelSwapComplete) {
          clearTimeout(timeoutId);
          resolve(true);
        }
      };
      
      // Add listener for future completion
      this.modelLoadListeners.push(() => {
        checkComplete();
      });
      
      // Check current state as well (in case it loaded between above check and here)
      checkComplete();
    });
  }

  private async loadBundledModelFromMemory(): Promise<tf.LayersModel | null> {
    try {
        console.log("Attempting manual bundled model load via expo-asset -> fromMemory...");
        const modelJson = MODEL_ASSETS.modelJson;
        const weightsAsset = Asset.fromModule(MODEL_ASSETS.modelWeightsModule);

        console.log("  Ensuring asset is downloaded...");
        if (!weightsAsset.downloaded) {
            await weightsAsset.downloadAsync();
        }
        const weightsUri = weightsAsset.localUri;
        if (!weightsUri) throw new Error("Asset download failed - no localUri.");
        console.log(`  Weights asset local URI: ${weightsUri}`);

        console.log("  Reading weights file as Base64...");
        const weightsBase64 = await FileSystem.readAsStringAsync(weightsUri, {
            encoding: FileSystem.EncodingType.Base64,
        });
        console.log(`  Weights file read successfully.`);

        console.log("  Decoding Base64 to ArrayBuffer...");
        const weightsArrayBuffer = Buffer.from(weightsBase64, 'base64').buffer;
        console.log(`  Decoded to ArrayBuffer (bytes: ${weightsArrayBuffer.byteLength}).`);

        console.log("  Extracting weight specs...");
        const weightSpecs = modelJson?.weightsManifest?.[0]?.weights || [];
        if (weightSpecs.length === 0) console.warn("  Warning: No weight specs found in model.json");
        else console.log(`  Found ${weightSpecs.length} weight specs.`);

        console.log("  Creating model artifacts object...");
        const modelArtifacts: tf.io.ModelArtifacts = {
            modelTopology: modelJson.modelTopology,
            weightSpecs: weightSpecs,
            weightData: weightsArrayBuffer,
            format: modelJson.format,
            generatedBy: modelJson.generatedBy,
            convertedBy: modelJson.convertedBy
        };

        console.log("  Loading model via tf.io.fromMemory(modelArtifacts)...");
        const model = await tf.loadLayersModel(tf.io.fromMemory(modelArtifacts));

        console.log("✅ Success: Loaded BUNDLED model via expo-asset -> fromMemory.");
        return model;

    } catch (error: any) {
        console.error("❌ Failed during manual bundled model load:", error);
        if (error.message) console.error("   Error message:", error.message);
        return null;
    }
  }

  async loadModel(): Promise<void> {
    // Prevent concurrent initial loading
    if (this.isLoading && !this.initialLoadComplete) {
        console.log("Initial model loading already in progress, waiting...");
        while (this.isLoading && !this.initialLoadComplete) {
            await new Promise(res => setTimeout(res, 100));
        }
        if (this.modelLoaded) {
            console.log("Initial model was loaded by another process.");
            return;
        }
    }

    if (this.modelLoaded) {
        console.log("Model is already loaded.");
        return;
    }

    this.isLoading = true;
    this.modelSwapComplete = false;
    console.log("Starting initial model loading sequence...");

    let modelLoadSuccess = false;

    // Step 1 & 2: Initialise TFJS and Load Model
    try {
        console.log("Initializing TensorFlow and verifying environment...");
        await tf.ready();
        await tf.setBackend('cpu');
        const backend = tf.getBackend();
        console.log(`  TensorFlow.js ready. Backend: ${backend}`);

        if (typeof tf.util?.isTypedArray !== 'function') {
            console.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            console.error("CRITICAL INIT FAILURE: tf.util.isTypedArray is UNDEFINED immediately after tf.ready()!");
            console.error("This points to a core TFJS/Platform setup issue.");
            console.error("Verify @tensorflow/tfjs, @tensorflow/tfjs-react-native versions and compatibility.");
            console.error("Perform a full clean install if versions seem correct.");
            console.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            throw new Error("TFJS failed basic initialization (tf.util missing).");
        } else {
            console.log("  Basic TFJS utilities (tf.util.isTypedArray) seem available.");
        }

        try {
            console.log(`Successfully set backend: ${tf.getBackend()}`);
        } catch (backendError) {
            console.warn(`Could not set 'rn-webgl' backend (using ${tf.getBackend()}):`, backendError);
        }

        // Step 2: Load model EXCLUSIVELY via manual memory method
        console.log("Attempting primary load: Manual expo-asset -> fromMemory method...");
        const loadedModel = await this.loadBundledModelFromMemory();

        // Step 3: Check if Loading Succeeded
        if (!loadedModel) {
             console.error("CRITICAL: Manual model loading from memory failed.");
             throw new Error("Failed to load model via loadBundledModelFromMemory.");
        }

        // Loading succeeded
        console.log("Model loading successful (using fromMemory).");
        this.model = loadedModel;
        this.isUsingTrainedModel = false;
        this.modelLoaded = true;
        modelLoadSuccess = true;
    } catch (error: any) {
        // This catches errors from tf.ready(), the init checks, or loadBundledModelFromMemory
        console.error("Unhandled error during TFJS init or main model load:", error);
        this.modelLoaded = false;
        this.isUsingTrainedModel = false;
        modelLoadSuccess = false;
    }

    // Always mark these states regardless of success/failure
    this.isLoading = false;
    this.initialLoadComplete = true;

    // Only proceed with optional checks if loading was successful
    if (modelLoadSuccess) {
        console.log("Initial bundled load step complete. Proceeding with checks...");

        // Optional: Try Test Prediction (Safely)
        try {
            // **** TEMPORARILY DISABLE the actual test call for now ****
            // await this.runTestPrediction("Initial Bundled Model (fromMemory)");
            console.log("Skipping initial test prediction for now."); // Log that we skipped it
        } catch (testPredError: any) {
            // This catch block is now only for the test prediction itself
            console.error("Error during initial test prediction (model is loaded, but test failed):", testPredError);
            // DO NOT set modelLoaded = false here
        }

        // Notify listeners about successful loading
        this.notifyModelLoadListeners();

        // --- STEP 4: Trigger Async Trained Model Check ---
        console.log("Triggering async trained model check...");
        
        // Use setTimeout to ensure it runs *after* the current execution context
        setTimeout(() => {
            this._attemptLoadTrainedModel()
                .catch(err => {
                    console.error("Error during async trained model load attempt:", err);
                })
                .finally(() => {
                    // Always mark the swap as complete, even if it failed
                    this.modelSwapComplete = true;
                    this.notifyModelLoadListeners();
                    console.log("Model load process complete (including trained model check)");
                });
        }, 0);
    } else {
        // If main loading failed, handle the failure state
        console.error("Model loading failed, skipping post-load checks.");
        this.modelSwapComplete = true; // Mark swap check as ended (failure)
        this.notifyModelLoadListeners(); // Notify about failure
    }
  }

  // Helper to notify listeners when model loading state changes
  private notifyModelLoadListeners(): void {
    if (this.modelLoadListeners.length > 0) {
      console.log(`Notifying ${this.modelLoadListeners.length} model load listeners`);
      
      // Call each listener and then clear the list
      for (const listener of this.modelLoadListeners) {
        try {
          listener();
        } catch (e) {
          console.error("Error in model load listener:", e);
        }
      }
    }
  }

  // --- Modify _attemptLoadTrainedModel to use the single-argument fromMemory approach ---
  private async _attemptLoadTrainedModel(): Promise<void> {
    console.log("Async Check: Attempting to load trained model manually from AsyncStorage...");
    try {
        const serializedModel = await AsyncStorage.getItem(MODEL_STORAGE_KEY);

        if (serializedModel) {
            console.log("   Async Check: Found serialized trained model.");
            const artifacts = JSON.parse(serializedModel);

            if (artifacts?.weightData && typeof artifacts.weightData === 'string') {
                console.log("   Async Check: Converting base64 weightData...");
                const weightDataBase64 = artifacts.weightData;
                const binary = Buffer.from(weightDataBase64, 'base64');
                const loadableArtifacts = {
                    modelTopology: artifacts.modelTopology,
                    weightSpecs: artifacts.weightSpecs,
                    weightData: binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength),
                    format: artifacts.format,
                    generatedBy: artifacts.generatedBy,
                    convertedBy: artifacts.convertedBy
                };

                console.log("   Async Check: Attempting tf.loadLayersModel(tf.io.fromMemory(...))");
                const trainedModel = await tf.loadLayersModel(tf.io.fromMemory(loadableArtifacts));
                console.log("✅ Async Check: Successfully loaded TRAINED model manually.");

                // SWAP THE MODELS
                // Dispose the old (bundled) model before assigning the new one
                if (this.model && this.model !== trainedModel) {
                    console.log("   Async Check: Disposing previous (bundled) model.");
                    this.model.dispose();
                }
                this.model = trainedModel;
                this.isUsingTrainedModel = true;
                this.isCompiledForTraining = false;

                await this.runTestPrediction("Async Loaded Trained Model");

                console.log("   Async Check: Model swap complete. Now using TRAINED model.");

            } else {
                console.warn("   Async Check: Serialized model found, but weightData format is incorrect or missing.");
            }
        } else {
            console.log("   Async Check: No trained model found in AsyncStorage. Continuing with bundled model.");
        }
    } catch (manualError: any) {
        // Don't crash the app, just log the error and continue using the bundled model
        console.error("Async Check: Failed to load trained model manually:", manualError.message || manualError);
        // We keep using the already loaded bundled model
    }
  }

  // Helper function to run test prediction
  private async runTestPrediction(modelSource: string = "unknown"): Promise<void> {
      if (!this.model) {
          console.error(`Cannot run test prediction for ${modelSource}: Model is null.`);
          throw new Error(`Model is null, cannot test prediction for ${modelSource}`);
      }

      console.log(`Running test prediction with ${modelSource}...`);
      // Use tf.tidy to manage test tensors automatically
      tf.tidy(() => {
          const batchSize = 1;
          const testItemFeatures = tf.zeros([batchSize, MAX_ITEMS, NEW_FEATURE_DIM]);
          const testItemMasks = tf.zeros([batchSize, MAX_ITEMS]);
          const testMainColors = tf.zeros([batchSize, MAX_ITEMS, COLOR_FEATURES_PER_ITEM]);
          const testTargetTemps = tf.zeros([batchSize, 1]);

          try {
              const res = this.model!.predict([
                  testItemFeatures,
                  testItemMasks,
                  testMainColors,
                  testTargetTemps
              ]) as tf.Tensor;

              console.log(`Test prediction successful for ${modelSource}.`);
          } catch (predError: any) {
              console.error(`Error during test prediction for ${modelSource}:`, predError);
              throw predError;
          }
      });
  }

  isModelLoaded(): boolean {
    return this.modelLoaded;
  }

  private _prepareInputs(
    outfitTensor: OutfitTensor,
    targetTemperature: number
  ): tf.Tensor[] {
    // For batch processing, although we're only doing one outfit at a time
    const batchSize = 1;
    
    // Initialise JavaScript arrays for tensors
    const jsItemFeatures = Array(batchSize * MAX_ITEMS * NEW_FEATURE_DIM).fill(0);
    const jsItemMasks = Array(batchSize * MAX_ITEMS).fill(0);
    const jsMainColors = Array(batchSize * MAX_ITEMS * COLOR_FEATURES_PER_ITEM).fill(0);
    const jsTargetTemps = Array(batchSize).fill(0);

    try {
      // Normalize context inputs
      jsTargetTemps[0] = Math.max(0.0, Math.min(1.0, targetTemperature / 4.0)) || 0;

      // Process outfit tensor (pad if needed)
      const processedTensor = [...outfitTensor.slice(0, MAX_ITEMS)];
      while (processedTensor.length < MAX_ITEMS) {
        processedTensor.push({});
      }

      for (let slotIdx = 0; slotIdx < MAX_ITEMS; slotIdx++) {
        const slot = processedTensor[slotIdx] || {};
        const slotFeatureBaseIndex = slotIdx * NEW_FEATURE_DIM;
        const slotColorBaseIndex = slotIdx * COLOR_FEATURES_PER_ITEM;

        // Handle item presence and mask
        let itemPresent = 0.0;
        if (slot.item_present !== undefined) {
          const parsedPresence = parseFloat(String(slot.item_present)) || 0.0;
          itemPresent = isNaN(parsedPresence) ? 0.0 : parsedPresence;
        }
        jsItemMasks[slotIdx] = itemPresent > 0 ? 1.0 : 0.0;

        if (itemPresent > 0) {
          // 1. PCA Values (76 features)
          let featureIdx = 0;
          const pcaValues = slot.pca_values || {};
          for (const pcaKey of Object.keys(pcaValues).sort()) {
            if (featureIdx < 76) {
              const valueStr = String(pcaValues[pcaKey] || 0);
              const value = parseFloat(valueStr);
              jsItemFeatures[slotFeatureBaseIndex + featureIdx] = isNaN(value) ? 0.0 : value;
              featureIdx++;
            }
          }
          
          // Fill remaining PCA slots with zeros
          while (featureIdx < 76) {
            jsItemFeatures[slotFeatureBaseIndex + featureIdx] = 0.0;
            featureIdx++;
          }

          // 2. Normalised Slot Index
          jsItemFeatures[slotFeatureBaseIndex + 76] = slotIdx / (MAX_ITEMS - 1);

          // 3. Normalised Item Temperature at position 77
          let itemTemperature = 2.0; // Default to Moderate
          
          // Try to get temperature from temperature_suitability array
          if (Array.isArray(slot.temperature_suitability) && slot.temperature_suitability.length > 0) {
            let sum = 0;
            let count = 0;
            for (const tempLevel of slot.temperature_suitability) {
              const parsedTemp = parseFloat(String(tempLevel));
              if (!isNaN(parsedTemp)) {
                sum += parsedTemp;
                count++;
              }
            }
            if (count > 0) {
              itemTemperature = sum / count;
            }
          } 
          // Fallback to single temperature value
          else if (slot.temperature !== undefined) {
            const parsedTemp = parseFloat(String(slot.temperature));
            if (!isNaN(parsedTemp)) {
              itemTemperature = parsedTemp;
            }
          }

          // Clamp between 0 and 4, then normalize
          const normalizedItemTemperature = Math.max(0.0, Math.min(4.0, itemTemperature)) / 4.0;
          jsItemFeatures[slotFeatureBaseIndex + 77] = normalizedItemTemperature;

          // 5. Process colors - with safer parsing
          const validColors: { percentage: number; hsv: [number, number, number] }[] = [];
          
          if (Array.isArray(slot.colors)) {
            for (const colorInfo of slot.colors) {
              if (colorInfo) {
                try {
                  // Safe parsing of percentage
                  let percentage = 0;
                  if (colorInfo.percentage !== undefined) {
                    const parsedPct = parseFloat(String(colorInfo.percentage));
                    percentage = isNaN(parsedPct) ? 0 : parsedPct;
                  }
                  
                  let hsv: [number, number, number] | undefined;
                  
                  // Try to get HSV directly
                  if (Array.isArray(colorInfo.hsv) && colorInfo.hsv.length === 3) {
                    const h = parseFloat(String(colorInfo.hsv[0])) || 0;
                    const s = parseFloat(String(colorInfo.hsv[1])) || 0;
                    const v = parseFloat(String(colorInfo.hsv[2])) || 0;
                    
                    hsv = [
                      h % 360,
                      Math.max(0, Math.min(1, s)),
                      Math.max(0, Math.min(1, v))
                    ];
                  } 
                  // Try to convert from RGB
                  else if (Array.isArray(colorInfo.rgb) && colorInfo.rgb.length === 3) {
                    const r = parseFloat(String(colorInfo.rgb[0])) || 0;
                    const g = parseFloat(String(colorInfo.rgb[1])) || 0;
                    const b = parseFloat(String(colorInfo.rgb[2])) || 0;
                    hsv = this.rgbToHsv(r, g, b);
                  }
                  
                  if (hsv) {
                    validColors.push({ percentage, hsv });
                  }
                } catch (e) {
                  console.warn("Invalid color entry:", e);
                }
              }
            }
          }
          
          // Sort colors by percentage (highest first)
          validColors.sort((a, b) => b.percentage - a.percentage);
          
          // Fill color features for top 3 colors with safer calculations
          for (let colorIdx = 0; colorIdx < 3; colorIdx++) {
            const colorOffset = colorIdx * 5;
            
            if (colorIdx < validColors.length) {
              const color = validColors[colorIdx];
              const percentageWeight = color.percentage / 100.0;
              const [h_degrees, s, v] = color.hsv;
              
              // Convert hue to radians with safe math
              const h_radians = (h_degrees || 0) * (Math.PI / 180.0);
              const sinH = Math.sin(h_radians) || 0;
              const cosH = Math.cos(h_radians) || 0;
              const weightedS = (s || 0) * percentageWeight;
              const weightedV = (v || 0) * percentageWeight;
              
              jsMainColors[slotColorBaseIndex + colorOffset + 0] = sinH;
              jsMainColors[slotColorBaseIndex + colorOffset + 1] = cosH;
              jsMainColors[slotColorBaseIndex + colorOffset + 2] = weightedS;
              jsMainColors[slotColorBaseIndex + colorOffset + 3] = weightedV;
              jsMainColors[slotColorBaseIndex + colorOffset + 4] = percentageWeight;
            }
          }
        }
      }

      // Use explicit typing and error handling for tensor creation
      const tensorItemFeatures = tf.tensor(jsItemFeatures, [batchSize, MAX_ITEMS, NEW_FEATURE_DIM], 'float32');
      const tensorItemMasks = tf.tensor(jsItemMasks, [batchSize, MAX_ITEMS], 'float32');
      const tensorMainColors = tf.tensor(jsMainColors, [batchSize, MAX_ITEMS, COLOR_FEATURES_PER_ITEM], 'float32');
      const tensorTargetTemps = tf.tensor(jsTargetTemps, [batchSize, 1], 'float32');

      // Return tensors in the order expected by the model
      return [tensorItemFeatures, tensorItemMasks, tensorMainColors, tensorTargetTemps];
    } catch (error) {
      console.error("Error preparing input tensors:", error);
      
      // Create safe fallback tensors in case of error
      const fallbackFeatures = tf.zeros([batchSize, MAX_ITEMS, NEW_FEATURE_DIM]);
      const fallbackMasks = tf.zeros([batchSize, MAX_ITEMS]);
      const fallbackColors = tf.zeros([batchSize, MAX_ITEMS, COLOR_FEATURES_PER_ITEM]);
      const fallbackTemps = tf.tensor([0.5], [batchSize, 1]);
      
      return [fallbackFeatures, fallbackMasks, fallbackColors, fallbackTemps];
    }
  }

  private rgbToHsv(r: number, g: number, b: number): [number, number, number] {
    r /= 255;
    g /= 255;
    b /= 255;
    
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    let h = 0;
    
    if (max === min) {
      h = 0;
    } else if (max === r) {
      h = 60 * ((g - b) / d + (g < b ? 6 : 0));
    } else if (max === g) {
      h = 60 * ((b - r) / d + 2);
    } else if (max === b) {
      h = 60 * ((r - g) / d + 4);
    }
    
    const s = max === 0 ? 0 : d / max;
    const v = max;
    
    return [h, s, v];
  }

  async predict(
    outfitTensor: OutfitTensor,
    targetTemperature: number
  ): Promise<number> {
    // Log the entire outfit tensor
    // console.log('====== COMPLETE OUTFIT TENSOR ======');
    // console.log(JSON.stringify(outfitTensor, null, 2));
    // console.log('===================================');

    // Wait if initial loading isn't complete yet
    if (this.isLoading && !this.initialLoadComplete) {
        console.warn('Predict called before initial model load finished. Waiting...');
        while (this.isLoading && !this.initialLoadComplete) {
            await new Promise(res => setTimeout(res, 100));
        }
        console.warn('Initial model load finished. Proceeding with predict...');
    }

    // If NO model loaded after waiting (critical error), return default
    if (!this.modelLoaded || !this.model) {
        console.error('Model unavailable for prediction.');
        console.error('modelJson', MODEL_ASSETS.modelJson);
        console.error('modelWeightsModule', MODEL_ASSETS.modelWeightsModule);
        console.error('modelLoaded', this.modelLoaded);

        return 0.5;
    }

    let inputs: tf.Tensor[] | null = null;
    let predictionTensor: tf.Tensor | null = null;
    
    try {
        console.log('Preparing input tensors...');
        inputs = this._prepareInputs(outfitTensor, targetTemperature);
        
        // **** DETAILED LOGGING HERE ****
        console.log("=== Input Tensors Before Predict ===");
        if (inputs && Array.isArray(inputs) && inputs.length === 4) {
             const [tensorItemFeatures, tensorItemMasks, tensorMainColors, tensorTargetTemps] = inputs;
             // Use optional chaining (?.) to safely access properties even if a tensor is null/undefined somehow
             console.log(`  ItemFeatures: Shape=${tensorItemFeatures?.shape}, DType=${tensorItemFeatures?.dtype}, Disposed=${tensorItemFeatures?.isDisposed}`);
             console.log(`  ItemMasks: Shape=${tensorItemMasks?.shape}, DType=${tensorItemMasks?.dtype}, Disposed=${tensorItemMasks?.isDisposed}`);
             console.log(`  MainColors: Shape=${tensorMainColors?.shape}, DType=${tensorMainColors?.dtype}, Disposed=${tensorMainColors?.isDisposed}`);
             console.log(`  TargetTemps: Shape=${tensorTargetTemps?.shape}, DType=${tensorTargetTemps?.dtype}, Disposed=${tensorTargetTemps?.isDisposed}`);
        } else {
             console.log(`  Inputs array structure is not as expected. Found: ${inputs === null ? 'null' : inputs === undefined ? 'undefined' : `length ${inputs.length}`}`);
        }
        console.log("===================================");
        
        // Enhanced logging to clearly show which model is being used for prediction
        console.log(`===== OUTFIT PREDICTION USING ${this.isUsingTrainedModel ? 'TRAINED' : 'BUNDLED'} MODEL =====`);
        predictionTensor = this.model!.predict(inputs) as tf.Tensor;
        
        if (!predictionTensor) {
            console.error('Prediction tensor is null');
            return 0.5; // Return neutral score on error
        }
        
        // Extract the raw value from the output tensor
        const scoreArray = await predictionTensor.data();
        if (!scoreArray || scoreArray.length === 0) {
            console.error('Empty prediction result');
            return 0.5; // Return neutral score on error
        }
        
        const score = scoreArray[0]; // Get the first value
        console.log(`TFJS prediction result: ${score} (using ${this.isUsingTrainedModel ? 'TRAINED' : 'BUNDLED'} model)`);
        
        // Ensure we return a valid number between 0 and 1
        if (isNaN(score)) {
            console.error('NaN prediction result, returning default 0.5');
            return 0.5;
        }
        
        return Math.max(0.0, Math.min(1.0, score)); // Clamp between 0 and 1
    } catch (error) {
        console.error('TFJS inference failed:');
        console.error('Error details:', error);
        if (error instanceof Error) {
            console.error('Stack trace:', error.stack);
        }
        return 0.5; // Return neutral score on error
    } finally {
        // Clean up tensors to prevent memory leaks
        try {
            if (inputs) {
                console.log('Disposing input tensors...');
                inputs.forEach(tensor => {
                    if (tensor && tensor.dispose) {
                        tensor.dispose();
                    }
                });
            }
            if (predictionTensor && predictionTensor.dispose) {
                console.log('Disposing prediction tensor...');
                predictionTensor.dispose();
            }
        } catch (disposeError) {
            console.error('Error during tensor cleanup:', disposeError);
        }
    }
  }

  async printScores(
    outfits: Array<{tensor: OutfitTensor, label?: string}>,
    targetTemperature: number = 2,
    printDetails: boolean = false
  ): Promise<number[]> {
    if (!this.modelLoaded) {
      await this.loadModel();
    }
    
    console.log('\n===== OUTFIT SCORES (TFJS) =====');
    console.log(`Target Temperature: ${targetTemperature}/4\n`);
    
    const scores: number[] = [];
    
    for (let i = 0; i < outfits.length; i++) {
      const { tensor, label } = outfits[i];
      const outfitId = label || `Outfit ${i + 1}`;
      
      try {
        const score = await this.predict(tensor, targetTemperature);
        scores.push(score);
        
        // Format score as percentage
        const scorePercent = (score * 100).toFixed(1);
        console.log(`${outfitId}: ${scorePercent}% match`);
        
        // Print detailed breakdown if requested
        if (printDetails) {
          console.log(`  Items: ${tensor.filter(slot => slot.item_present === 1).length}`);
          
          tensor.forEach((slot, idx) => {
            if (slot.item_present) {
              console.log(`  [${idx}] Item ${idx + 1}`);
              
              if (slot.temperature_suitability && slot.temperature_suitability.length > 0) {
                console.log(`    Temperature: ${slot.temperature_suitability.join(', ')}`);
              } else if (slot.temperature !== undefined) {
                console.log(`    Temperature: ${slot.temperature}`);
              }
              
              if (slot.colors && slot.colors.length > 0) {
                const mainColors = slot.colors
                  .slice(0, 2)
                  .map(c => {
                    const rgb = c.rgb ? `RGB(${c.rgb.join(',')})` : '';
                    const pct = c.percentage ? `${c.percentage}%` : '';
                    return `${rgb} ${pct}`.trim();
                  })
                  .join(', ');
                console.log(`    Colors: ${mainColors}`);
              }
            }
          });
          console.log('');
        }
      } catch (error) {
        console.error(`Error scoring ${outfitId}:`, error);
        scores.push(0);
      }
    }
    
    console.log('================================\n');
    return scores;
  }

  close(): void {
    if (this.model) {
      console.log('Disposing TFJS Recommender model resources');
      
      try {
        // Dispose the model's resources but keep the saved model in AsyncStorage
        this.model.dispose();
        this.model = null;
        this.modelLoaded = false;
        this.isCompiledForTraining = false;
      } catch (error) {
        console.error('Error disposing model resources:', error);
      }
    }
  }

  private async compileModelForTraining(): Promise<boolean> {
    if (this.isCompiledForTraining) return true;
    if (!this.model || !this.modelLoaded) {
      console.error('Cannot compile: Model not loaded.');
      return false;
    }

    // Store original backend to potentially restore later
    const originalBackend = tf.getBackend();
    
    try {
      try {
        console.log(`Current backend before training: ${originalBackend}`);
        await tf.setBackend('cpu'); // Try switching before compile
        console.warn(`>>>> Switched to CPU backend FOR TRAINING. Current: ${tf.getBackend()} <<<<`);
      } catch (cpuErr) {
        console.error("Failed to switch to CPU backend for training, continuing with default.", cpuErr);
        // Don't return false here, maybe the default works? But log it.
      }

      console.log('Compiling model for training...');
      const optimizer = tf.train.adam(0.0015);
      this.model.compile({
        optimizer: optimizer,
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
      });
      this.isCompiledForTraining = true;
      console.log('Model compiled successfully for training.');
      return true;
    } catch (error) {
      console.error('Failed to compile model for training:', error);
      this.isCompiledForTraining = false;
      
      // Try to restore original backend if compilation failed
      if (tf.getBackend() !== originalBackend) {
        try {
          console.log(`Restoring original backend (${originalBackend}) after failed compilation`);
          await tf.setBackend(originalBackend);
        } catch (restoreErr) {
          console.error("Failed to restore original backend:", restoreErr);
        }
      }
      
      return false;
    }
  }

  async train(
    outfitTensors: OutfitTensor[],
    labels: number[],
    temps: number[],
    {
      epochs = 10,
      batchSize = 16,
      validationSplit = 0.1,
      classWeights,
      validationData,    // { tensors, labels, temps }
    }: {
      epochs?: number,
      batchSize?: number,
      validationSplit?: number,
      classWeights?: { [label: number]: number },
      validationData?: {
        tensors: OutfitTensor[],
        labels: number[],
        temps: number[]
      }
    } = {}
  ): Promise<tf.History> {
    if (!this.model || !this.modelLoaded) {
      throw new Error("Model must be loaded before training");
    }
    if (
      outfitTensors.length !== labels.length ||
      labels.length !== temps.length
    ) {
      throw new Error("Inputs, labels and temps must have same length");
    }

    // Log current memory state and TFJS backend
    console.log("======= TRAINING DEBUG INFORMATION =======");
    console.log(`TFJS version: ${tf.version.tfjs}`);
    console.log(`Backend in use: ${tf.getBackend()}`);
    console.log(`Memory before training:`, tf.memory());
    console.log(`Training dataset size: ${outfitTensors.length} samples`);
    console.log(`Batch size: ${batchSize}, Epochs: ${epochs}`);
    console.log(`Validation split: ${validationSplit}`);
    console.log(`Using external validation data: ${validationData ? 'Yes' : 'No'}`);
    console.log("==========================================");

    // Add validation for input tensors
    console.log(`Validating ${outfitTensors.length} tensors for training...`);
    for (let i = 0; i < outfitTensors.length; i++) {
      const tensor = outfitTensors[i];
      if (!tensor || !Array.isArray(tensor) || tensor.length === 0) {
        throw new Error(`Invalid tensor at index ${i}: ${JSON.stringify(tensor)}`);
      }
      // Validate each slot in the tensor
      for (let j = 0; j < tensor.length; j++) {
        if (!tensor[j]) {
          console.warn(`Missing slot at index ${j} in tensor ${i}, replacing with empty object`);
          tensor[j] = {};
        }
      }
    }

    // Ensure model is compiled for training
    await this.compileModelForTraining();

    // Use a structured approach to tensor management
    let xFeat: tf.Tensor = null!;
    let xMask: tf.Tensor = null!;
    let xCol: tf.Tensor = null!;
    let xTemp: tf.Tensor = null!;
    let y: tf.Tensor = null!;
    let valTensors: Array<tf.Tensor> = [];
    
    try {
      // Let's simplify the entire approach to avoid shape issues completely
      console.log("Building tensors directly from input data...");
      
      // Use tidy to ensure proper memory management
      [xFeat, xMask, xCol, xTemp] = tf.tidy(() => {
        // Create empty arrays of tensors
        const featureTensors: tf.Tensor[] = [];
        const maskTensors: tf.Tensor[] = [];
        const colorTensors: tf.Tensor[] = [];
        const tempTensors: tf.Tensor[] = [];
        
        // Process each outfit
        for (let i = 0; i < outfitTensors.length; i++) {
          // Get the tensors for this outfit
          const [f, m, c, t] = this._prepareInputs(outfitTensors[i], temps[i]);
          
          // Add to our collections
          featureTensors.push(f);
          maskTensors.push(m);
          colorTensors.push(c);
          tempTensors.push(t);
        }
        
        // Stack all tensors along batch dimension
        const stackedFeatures = tf.concat(featureTensors, 0);
        const stackedMasks = tf.concat(maskTensors, 0);
        const stackedColors = tf.concat(colorTensors, 0);
        const stackedTemps = tf.concat(tempTensors, 0);
        
        return [stackedFeatures, stackedMasks, stackedColors, stackedTemps];
      });
      
      // Create labels tensor directly
      y = tf.tensor2d(labels, [labels.length, 1], 'float32');
      
      // 3) Callbacks - determine if we will have validation data
      const hasValidation = validationData || (labels.length >= 20 && validationSplit > 0);
      const callbacks: tf.Callback[] = [];
      
      // Only add callbacks that monitor validation metrics if we'll have validation data
      if (hasValidation) {
        // Use a simpler callback approach to avoid tensor issues
        callbacks.push(
          tf.callbacks.earlyStopping({
            monitor: 'val_loss',
            patience: 5
          })
        );
      }

      // 4) Build fit config
      const fitConfig: tf.ModelFitArgs = {
        epochs,
        batchSize: Math.min(batchSize, labels.length),
        shuffle: true,
        callbacks,
        classWeight: classWeights,
      };

      if (validationData) {
        // Create validation tensors using the same direct approach
        const vxFeat = tf.tidy(() => {
          // Create tensor arrays
          const vFeatureTensors: tf.Tensor[] = [];
          const vMaskTensors: tf.Tensor[] = [];
          const vColorTensors: tf.Tensor[] = [];
          const vTempTensors: tf.Tensor[] = [];
          
          // Process each validation outfit
          for (let i = 0; i < validationData.tensors.length; i++) {
            // Get tensors
            const [vf, vm, vc, vt] = this._prepareInputs(
              validationData.tensors[i],
              validationData.temps[i]
            );
            
            // Add to collections
            vFeatureTensors.push(vf);
            vMaskTensors.push(vm);
            vColorTensors.push(vc);
            vTempTensors.push(vt);
          }
          
          // Stack tensors
          const stackedVFeatures = tf.concat(vFeatureTensors, 0);
          const stackedVMasks = tf.concat(vMaskTensors, 0);
          const stackedVColors = tf.concat(vColorTensors, 0);
          const stackedVTemps = tf.concat(vTempTensors, 0);
          
          return [stackedVFeatures, stackedVMasks, stackedVColors, stackedVTemps];
        });
        
        // Create validation labels tensor directly
        const vLabels = tf.tensor2d(validationData.labels, [validationData.labels.length, 1]);
        
        // Store validation tensors for later disposal
        valTensors = [...vxFeat, vLabels];
        
        fitConfig.validationData = [
          vxFeat,
          vLabels
        ];
      } else if (labels.length >= 20) {
        // fallback to split
        fitConfig.validationSplit = validationSplit;
      }

      // 5) Train
      console.log(`Starting training for ${epochs} epochs with backend: ${tf.getBackend()}…`);
      console.log("Memory before fit:", tf.memory());
      
      // Run the fit operation
      console.log("CRITICAL POINT: About to call model.fit() - watch for reshape errors here");
      const history = await this.model!.fit(
        [xFeat, xMask, xCol, xTemp],
        y,
        fitConfig
      );
      
      console.log("SUCCESS: Training finished without errors");
      console.log("Memory after fit:", tf.memory());
      
      return history;
      
    } catch (error: any) {
      console.error("==== TRAINING ERROR DETAILS ====");
      console.error(`Current backend: ${tf.getBackend()}`);
      console.error("Error type:", error.constructor.name);
      console.error("Training error message:", error.message);
      if (error.stack) {
        console.error("Error stack trace:", error.stack);
      }
      console.error("================================");
      throw error;
    } finally {
      // Always clean up tensors
      console.log("Cleaning up training tensors...");
      
      // Dispose training tensors
      [xFeat, xMask, xCol, xTemp, y].forEach(tensor => {
        if (tensor && !tensor.isDisposed) {
          try {
            tensor.dispose();
          } catch (e) {
            console.warn("Error disposing tensor:", e);
          }
        }
      });
      
      // Dispose validation tensors
      valTensors.forEach(tensor => {
        if (tensor && !tensor.isDisposed) {
          try {
            tensor.dispose();
          } catch (e) {
            console.warn("Error disposing validation tensor:", e);
          }
        }
      });
      
      console.log("Final memory state:", tf.memory());
    }
  }

  // Save the trained model to AsyncStorage
  async saveModel(): Promise<boolean> {
    if (!this.model || !this.modelLoaded) {
      console.error("Cannot save model: No model loaded");
      return false;
    }
    
    try {
      console.log("Saving trained model to AsyncStorage...");
      
      // Ensure TensorFlow.js is fully initialized
      await tf.ready();
      
      // Try the simplest approach - use the direct IO handler
      try {
        const modelPath = `rn-async-storage://${MODEL_STORAGE_KEY}`;
        await this.model.save(modelPath);
        console.log("Model saved successfully using built-in IO handler");
      } catch (saveError) {
        console.log("Built-in save failed:", saveError);
        console.log("Falling back to manual save with base64 encoding");
        
        // Fallback to manual save
        const saveResult = await this.model.save(tf.io.withSaveHandler(async (artifacts) => {
          // Need to convert weightData (ArrayBuffer) to a base64 string
          const weightData = artifacts.weightData as ArrayBuffer;
          const uint8Array = new Uint8Array(weightData);
          const base64Data = Buffer.from(uint8Array).toString('base64');
          
          // Create storable object
          const storableArtifacts = {
            modelTopology: artifacts.modelTopology,
            weightSpecs: artifacts.weightSpecs,
            weightData: base64Data,
            format: artifacts.format,
            convertedBy: artifacts.convertedBy
          };
          
          // Store as JSON string
          await AsyncStorage.setItem(MODEL_STORAGE_KEY, JSON.stringify(storableArtifacts));
          console.log("Model saved to AsyncStorage with manual base64 encoding");
          
          return {
            modelArtifactsInfo: {
              dateSaved: new Date(),
              modelTopologyType: 'JSON',
              weightDataBytes: weightData.byteLength
            }
          };
        }));
        
        console.log("Manual save complete:", saveResult);
      }
      
      // Set flag to indicate we're using a trained model now
      this.isUsingTrainedModel = true;
      console.log("Model status updated: Using trained model");
      return true;
    } catch (error) {
      console.error("Error saving model to AsyncStorage:", error);
      return false;
    }
  }
  
  // Method to delete the saved model
  async deleteSavedModel(): Promise<void> {
    try {
      await tf.ready();
      
      // Try to delete using built-in handler
      try {
        const modelPath = `rn-async-storage://${MODEL_STORAGE_KEY}`;
        await tf.io.removeModel(modelPath);
        console.log("Model deleted using built-in handler");
      } catch (error) {
        console.log("Built-in delete failed, using direct AsyncStorage removal");
      }
      
      // Also try direct AsyncStorage removal as backup
      await AsyncStorage.removeItem(MODEL_STORAGE_KEY);
      console.log("Model deleted from AsyncStorage");
      
      this.isUsingTrainedModel = false;
      console.log("Model status updated: Using original model");
    } catch (error) {
      console.error("Error deleting saved model:", error);
    }
  }
  
  // Method to check if we're using a trained model
  isUsingPersistedModel(): boolean {
    return this.isUsingTrainedModel;
  }
  
  async resetTensorMemory(): Promise<void> {
    console.log("===== STARTING TENSOR MEMORY RESET =====");
    
    // Add timeout safety to prevent hanging
    const resetTimeout = setTimeout(() => {
      console.error("RESET TIMEOUT: Operation took too long, forcing reload");
      this.modelLoaded = false;
      this.isUsingTrainedModel = false;
    }, 10000); // 10 second timeout
    
    try {
      // Step 1: Remove existing model from storage
      console.log("RESET STEP 1: Deleting saved model...");
      await this.deleteSavedModel();
      console.log("✓ Model deleted from storage");
      
      // Step 2: Clean up existing model
      console.log("RESET STEP 2: Disposing existing model...");
      if (this.model) {
        this.model.dispose();
        this.model = null;
      }
      this.modelLoaded = false;
      this.isCompiledForTraining = false;
      this.isUsingTrainedModel = false;
      console.log("✓ Model disposed");
      
      // Step 3: Clean up any lingering tensors
      console.log("RESET STEP 3: Running garbage collection...");
      
      // Be very careful with memory cleanup to avoid errors
      try {
        console.log(`Before GC: ${tf.memory().numTensors} tensors in memory`);
        
        // Use only the most reliable cleaning methods
        tf.disposeVariables();
        
        // Safer approach to tensor cleanup
        const tensors = tf.engine().state.numTensors;
        if (tensors > 0) {
          console.log(`Found ${tensors} tensors to clean up`);
        }
        
        // Add a small delay to let async cleanup complete
        await new Promise(resolve => setTimeout(resolve, 200));
        
        console.log(`After safe GC: ${tf.memory().numTensors} tensors in memory`);
        console.log("✓ Memory cleaned with safe approach");
      } catch (gcError) {
        // If we get an error during GC, log it but continue
        console.warn("Non-critical error during garbage collection:", gcError);
        console.log("Continuing with reset despite GC error");
      }
      
      // Step 4: Reload fresh bundled model
      console.log("RESET STEP 4: Loading bundled model...");
      
      try {
        // Make sure TF is ready
        await tf.ready();
       
        this.model = await this.loadBundledModelFromMemory();
        if (!this.model) {
          throw new Error("Failed to load model via loadBundledModelFromMemory during reset");
        }
        
        this.modelLoaded = true;
        console.log("✓ Bundled model loaded successfully using loadBundledModelFromMemory");
      } catch (bundleError) {
        console.error("Error loading bundled model:", bundleError);
        throw bundleError;
      }
      
      console.log("===== RESET COMPLETE =====");
      console.log("Final tensor count:", tf.memory().numTensors);
    } catch (error) {
      console.error("Error during tensor memory reset:", error);
      
      // Try a simpler fallback approach if the detailed steps failed
      try {
        console.log("Attempting simplified reset fallback...");
        
        // Simple cleanup
        if (this.model) {
          this.model.dispose();
          this.model = null;
        }
        
        tf.disposeVariables();
        this.modelLoaded = false;
        this.isUsingTrainedModel = false;
        
        // Load using our new method
        await tf.ready();
        this.model = await this.loadBundledModelFromMemory();
        if (!this.model) {
          throw new Error("Fallback model load also failed");
        }
        this.modelLoaded = true;
        
        console.log("Fallback reset completed");
      } catch (fallbackError) {
        console.error("Fallback reset also failed:", fallbackError);
        throw new Error("Complete reset failure. App restart required.");
      }
    } finally {
      // Make sure we clear the timeout
      clearTimeout(resetTimeout);
    }
  }
}
