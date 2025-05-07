/**
 * This module implements a full outfit recommendation system using a TensorFlow.js model.
 *
 * - TFJSScorer: Adapts a TFJSRecommender to the generic OutfitScorer interface, handling model initialisation,
 *   prediction (sync fallback and async strategies), caching, and fallback scoring when the model isn't ready.
 *
 * - OutfitGenerator: Loads and preprocesses wardrobe items, converts them into tensor representations,
 *   and generates outfits via hill climbing or beam search based on a target temperature.
 *   Supports both 'generate' and 'swipe' modes with optional precomputed outfits.
 *
 * - generateOutfit: High-level function to initialise the scorer and generator, wait for the model,
 *   and produce the best outfit according to the selected strategy.
*/

import { TFJSRecommender } from './TFJSRecommender';

export interface OutfitScorer {
  scoreOutfitTensor(tensor: OutfitTensor, targetTemperature: number, targetFormality?: number): number;
}

export interface ITFJSRecommender {
  loadModel(): Promise<void>;
  predict(tensor: OutfitTensor, targetTemperature: number): Promise<number>;
  close(): void;
  getActiveModelType?(): string;
  onModelSwap?(callback: (modelType: string) => void): void;
}

export interface WardrobeItem {
    id?: string;               
    uri?: string;             
    processedUri?: string;    
    class_name: string;
    clothing_category: string;
    clothing_category_index: string | number;
    confidence: string | number;
    description?: string | null;
    pca_values?: Record<string, number>;
    principalComponents?: number[]; 
    colors?: Array<[string, number, number[]]> | (string | number)[] | Array<{name: string, rgb: number[], percentage: number}>;
    filename?: string;
    segmented_image?: string;
    bounding_box?: [number, number, number, number] | null;
    bbox?: number[] | null;
    item_id?: string;
    category?: string;          
    className?: string;         
    context?: any;
    gender?: any;
    temperature_suitability?: number[];
    temperature?: number;       
    seasons?: string[];
}

export interface ColorInfo {
    rgb?: [number, number, number];
    hsv?: [number, number, number];
    percentage: number | string;
}

interface ItemsByClass {
    [className: string]: WardrobeItem[];
}

interface LayerFlags {
    use_outer: boolean;
}

export interface OutfitTensorSlot {
    item_present: number;
    class_name: string | null;
    clothing_category: string;
    clothing_category_index: string;
    confidence: string | number;
    description: string | null;
    pca_values: Record<string, number> | {};
    colors: ColorInfo[];
    temperature_suitability?: number[];
    seasons?: string[];
    filename?: string | null;
    segmented_image_filename?: string | null;
    item_id?: string;
    uri?: string;
    processedUri?: string;
    id?: string;
    segmented_image?: string | null; 
}

export type OutfitTensor = OutfitTensorSlot[];

export type Outfit = {
    outer: WardrobeItem | null;
    base: WardrobeItem | null;
    bottom: WardrobeItem | null;
};


const NUM_OUTFITS_TO_GENERATE = 5;

// First mapping: Categories to DeepFashion2 classes
const CATEGORY_TO_CLASSES: Record<string, string[]> = {
  "T-shirts/Tanks": ["short_sleeve_top", "sling"],
  "Shirts": ["long_sleeve_top"],
  "Sweaters/Hoodies": ["long_sleeve_top"],
  "Light Outerwear": ["long_sleeve_outwear"],
  "Heavy Outerwear": ["long_sleeve_outwear"],
  "Trousers": ["trousers"],
  "Shorts": ["shorts"],
  "Skirts": ["skirt"],
  "Athletic Bottoms": ["shorts", "trousers"],
  "Dresses": ["short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"],
  "Jumpsuits": ["short_sleeve_dress", "long_sleeve_dress"],
  "Formal Wear": ["long_sleeve_top", "trousers", "long_sleeve_dress"],
  "Misc": []
};

const CLOTHING_CATEGORIES: Record<string, string[]> = {
  outer: ["long_sleeve_outwear", "short_sleeve_outwear", "long_sleeve_top"],
  base: ["vest", "short_sleeve_top", "sling"],
  bottom: ["trousers", "shorts", "skirt"]
};

const SLOT_ORDER = ["outer", "base", "bottom"];

function determineItemClass(item: WardrobeItem): string {
  const innateClass = item.class_name || item.className;
  const userCategory = item.clothing_category || item.category;
  
  // Priority 1: If DeepFashion2 classifies as any type of outerwear, respect that classification
  if (innateClass && (innateClass === "long_sleeve_outwear" || innateClass === "short_sleeve_outwear")) {
    return innateClass;
  }
  
  // Priority 2: If user categorised as Heavy Outerwear, force it to be long_sleeve_outwear
  if (userCategory === "Heavy Outerwear") {
    return "long_sleeve_outwear";
  }
  
  // Check if user category exists in our mapping
  if (!userCategory || !CATEGORY_TO_CLASSES[userCategory]) {
    return innateClass || "unknown";
  }
  
  const categoryDerivedClasses = CATEGORY_TO_CLASSES[userCategory];
  
  // Priority 3: If innate class aligns with user category, use it
  if (innateClass && categoryDerivedClasses.includes(innateClass)) {
    return innateClass;
  }
  
  // Priority 4: Fall back to first class from user category mapping
  return categoryDerivedClasses.length > 0 ? categoryDerivedClasses[0] : (innateClass || "unknown");
}

// TFJS Scorer Adapter - Implements OutfitScorer interface but uses TFJSRecommender
export class TFJSScorer implements OutfitScorer {
    private tfjsRecommender: TFJSRecommender;
    private isModelLoaded: boolean = false;
    private isModelLoading: boolean = false;
    private lastPredictionCache: Map<string, number> = new Map();
    private activeModelType: string = 'BUNDLED'; // Track which model is active
    private isInitialized: boolean = false;
    private initPromise: Promise<void> | null = null;

    constructor() {
        this.tfjsRecommender = new TFJSRecommender();
        this.initPromise = this.initialize();
    }

    // Create a proper async initialization method
    private async initialize(): Promise<void> {
        if (this.isInitialized) return;
        
        try {
            this.isModelLoading = true;
            console.log("TFJSScorer: Starting complete model initialization...");
            
            const loadSuccess = await this.tfjsRecommender.waitForCompleteModelLoad(15000);
            
            if (loadSuccess) {
                this.isModelLoaded = true;
                console.log("TFJSScorer: Complete model initialization successful");
            } else {
                console.warn("TFJSScorer: Model load completed with issues - will use fallback if needed");
                // Still set model as loaded so we can try using it
                this.isModelLoaded = true;
            }
        } catch (error) {
            console.error("TFJSScorer initialization failed:", error);
            this.isModelLoaded = false;
        } finally {
            this.isModelLoading = false;
            this.isInitialized = true;
        }
    }

    // Make this public so UI can wait for initialization
    public async waitForInitialization(): Promise<boolean> {
        if (this.isInitialized) return this.isModelLoaded;
        if (this.initPromise) {
            await this.initPromise;
            return this.isModelLoaded;
        }
        return false;
    }

    // New async method that directly uses TFJSRecommender.predict
    async predictAsync(tensor: OutfitTensor, targetTemperature: number): Promise<number> {
        if (!this.isInitialized) {
            console.log("TFJSScorer: Waiting for model initialization before prediction...");
            await this.waitForInitialization();
        }
        
        if (!this.isModelLoaded) {
            console.log(`Using FALLBACK scoring method (model not loaded)`);
            return this.fallbackScoringMethod(tensor, targetTemperature);
        }

        try {
            console.log(`--- Generating Outfit prediction ---`);
            // Wait for the model prediction
            const score = await this.tfjsRecommender.predict(tensor, targetTemperature);
            
            // Cache the result for potential future synchronous calls
            const cacheKey = this.createCacheKey(tensor, targetTemperature);
            this.lastPredictionCache.set(cacheKey, score);
            
            console.log(`Outfit prediction complete`);
            return score;
        } catch (error) {
            console.error('TFJS async prediction failed:', error);
            return this.fallbackScoringMethod(tensor, targetTemperature);
        }
    }

    // Implement the OutfitScorer interface method
    scoreOutfitTensor(tensor: OutfitTensor, targetTemperature: number): number {
        // If initialisation hasn't completed, use fallback immediately
        if (!this.isInitialized || !this.isModelLoaded) {
            // Start initialisation if not already in progress
            if (!this.initPromise) {
                this.initPromise = this.initialize();
            }
            console.log(`Using FALLBACK scoring method (model not initialized)`);
            return this.fallbackScoringMethod(tensor, targetTemperature);
        }

        try {
            console.log(`--- Generating Outfit prediction ---`);
            // Create a cache key from outfit tensor + context
            const cacheKey = this.createCacheKey(tensor, targetTemperature);
            
            // Check if we have a cached prediction
            if (this.lastPredictionCache.has(cacheKey)) {
                const cachedScore = this.lastPredictionCache.get(cacheKey);
                console.log(`Using cached prediction`);
                return cachedScore || 0;
            }

            // Calculate a fallback score to return immediately
            const fallbackScore = this.fallbackScoringMethod(tensor, targetTemperature);
            
            // Schedule the TFJS prediction for next time (don't wait for it)
            setTimeout(() => {
                this.tfjsRecommender.predict(tensor, targetTemperature)
                    .then(score => {
                        // Cache the prediction for future use
                        this.lastPredictionCache.set(cacheKey, score);
                        console.log(`Cached new prediction for future use`);
                    })
                    .catch(error => {
                        console.error('TFJS prediction failed:', error);
                    });
            }, 0);
            
            // Return the fallback score for now
            return fallbackScore;
        } catch (error) {
            console.error('Error during TFJS scoring:', error);
            return this.fallbackScoringMethod(tensor, targetTemperature);
        }
    }

    // Create a simple cache key from the tensor and context
    private createCacheKey(tensor: OutfitTensor, temperature: number): string {
        // Extract item IDs from tensor and join with context values
        const itemIds = tensor.map(slot => slot.item_id || slot.id || 'empty').join('-');
        return `${itemIds}-t${temperature}`;
    }

    // Fallback scoring method when TFJS is not available
    private fallbackScoringMethod(tensor: OutfitTensor, targetTemperature: number): number {
        let score = 0;
        
        // Count items present in the outfit
        const presentItems = tensor.filter(slot => slot.item_present === 1.0);
        if (presentItems.length === 0) return 0;
        
        // Determine the outermost layer - first check for 'outer' layer, then 'base'
        let outermostLayerIdx = -1;
        if (tensor[0].item_present === 1.0) {
            outermostLayerIdx = 0; // Outer layer exists
        } else if (tensor[1].item_present === 1.0) {
            outermostLayerIdx = 1; // Base layer is outermost
        }
        
        // Score each present item
        for (let i = 0; i < presentItems.length; i++) {
            const item = presentItems[i];
            let itemScore = 0;
            
            // Temperature match - closer is better with heavy weight for exact match
            if (item.temperature_suitability?.length) {
                if (item.temperature_suitability.includes(targetTemperature)) {
                    itemScore += 0.5; // Exact match gets high score
                } else {
                    // Find closest temperature to target
                    const closest = item.temperature_suitability.reduce((prev, curr) => 
                        Math.abs(curr - targetTemperature) < Math.abs(prev - targetTemperature) ? curr : prev
                    );
                    // Partial score based on closeness (0.3 max for close, 0.1 for furthest)
                    itemScore += 0.3 * (1 - Math.abs(closest - targetTemperature) / 4);
                }
            }
            
            // Apply outermost layer bonus (double weighting for temperature match)
            // This makes the outermost layer's temperature suitability more important
            const isOutermostLayer = (tensor.indexOf(item) === outermostLayerIdx);
            if (isOutermostLayer) {
                itemScore *= 2.0; // Double the importance of the outermost layer
            }
            
            // Add small random factor for variety
            itemScore += Math.random() * 0.1;
            score += itemScore;
        }
        
        return score;
    }

    // Update model type - can be called from outside
    updateActiveModelType(modelType: string): void {
        this.activeModelType = modelType;
        console.log(`TFJSScorer: Active model type changed to ${modelType}`);
    }
    
    // Get current model type
    getActiveModelType(): string {
        return this.activeModelType;
    }

    // Clean up resources when done
    close(): void {
        if (this.isModelLoaded) {
            this.tfjsRecommender.close();
            this.isModelLoaded = false;
            this.lastPredictionCache.clear();
        }
    }
}

import wardrobeData from '../assets/precomputed_wardrobe.json';

// Add a mode type
export type OutfitGeneratorMode = 'swipe' | 'generate';

// Add interface for precomputed outfits
export interface PrecomputedOutfit {
  outfit: Outfit;
  tensor: OutfitTensor;
}

// Helper to normalize temperature value
function normalizeTemperature(item: WardrobeItem): number[] {
    // If it has temperature_suitability array, use it
    if (item.temperature_suitability && Array.isArray(item.temperature_suitability)) {
        return item.temperature_suitability;
    }
    
    // If it has a single temperature value (WardrobeScreen format)
    if (typeof item.temperature === 'number') {
        return [item.temperature];
    }
    
    // Default to medium temperature if no information
    return [2]; // 2 is "Moderate" in the new scale (0=Very Cold, 4=Hot)
}

// --- Outfit Generator Class ---

// Make sure to export the class
export class OutfitGenerator {
    private wardrobeItems: WardrobeItem[] = [];
    private itemsByClass: ItemsByClass = {};
    private precomputedOutfits: PrecomputedOutfit[] = [];
    private currentPrecomputedIndex: number = 0;
    private mode: OutfitGeneratorMode;

    constructor(
        wardrobeData: WardrobeItem[], 
        private outfitScorer: OutfitScorer = new TFJSScorer(),
        mode: OutfitGeneratorMode = 'generate',
        precomputedOutfits: PrecomputedOutfit[] = []
    ) {
        this.mode = mode;
        
        if (!wardrobeData || wardrobeData.length === 0) {
            console.warn("OutfitGenerator initialized with empty or invalid wardrobe data.");
            this.wardrobeItems = [];
            this.itemsByClass = {};
        } else {
            this.loadWardrobeItems(wardrobeData);
        }
        
        if (mode === 'swipe' && precomputedOutfits.length > 0) {
            this.precomputedOutfits = precomputedOutfits;
            console.log(`Outfit Generator Initialized (Swipe Mode with ${precomputedOutfits.length} outfits)`);
        } else {
            console.log(`Outfit Generator Initialized (${mode === 'swipe' ? 'Swipe' : 'Generate'} Mode)`);
        }
        
        console.log("Using TFJS recommendation model for outfit scoring");
    }

    private loadWardrobeItems(wardrobeData: WardrobeItem[]): void {
        try {
            // Pre-process items to determine their correct classes
            this.wardrobeItems = wardrobeData.map(item => {
                // Determine the correct class upfront
                const determinedClass = determineItemClass(item);
                
                // Create a new item with the determined class
                return {
                    ...item,
                    class_name: determinedClass,
                    // Also store original class for reference if needed
                    original_class_name: item.class_name || item.className
                };
            });

            // Organize items by class
            this.itemsByClass = {};
            for (const item of this.wardrobeItems) {
                // Use the already determined class
                const className = item.class_name || "unknown";
                if (!this.itemsByClass[className]) {
                    this.itemsByClass[className] = [];
                }
                this.itemsByClass[className].push(item);
            }

            console.log(`Processed ${this.wardrobeItems.length} wardrobe items`);
            // Log class distribution
            for (const [className, items] of Object.entries(this.itemsByClass)) {
                console.log(`  - ${className}: ${items.length} items`);
            }

        } catch (error) {
            console.error(`Error processing wardrobe items:`, error);
            // Reset state on error
            this.wardrobeItems = [];
            this.itemsByClass = {};
            throw new Error("Failed to process wardrobe items.");
        }
    }

    // Add helper for converting item to tensor slot
    private itemToTensorSlot(item: WardrobeItem): OutfitTensorSlot {
        // Store the best available image path with fallback mechanism
        let segmentedImageFilename: string | null = null;

        // Try each possible image source in order of preference
        if (item.segmented_image) {
            segmentedImageFilename = item.segmented_image;
            // console.log(`Using segmented_image for ${item.class_name || item.className || 'item'}: ${segmentedImageFilename}`);
        } else if (item.processedUri) {
            segmentedImageFilename = item.processedUri;
            // console.log(`Using processedUri for ${item.class_name || item.className || 'item'}: ${segmentedImageFilename}`);
        } else if (item.uri) {
            segmentedImageFilename = item.uri;
            // console.log(`Using uri for ${item.class_name || item.className || 'item'}: ${segmentedImageFilename}`);
        } else {
            console.warn(`No image path found for ${item.class_name || item.className || 'item'}`);
        }

        // Convert values to appropriate types for the tensor
        const confidenceValue = typeof item.confidence === 'string' 
            ? parseFloat(item.confidence) 
            : item.confidence;
        
        const categoryIndex = typeof item.clothing_category_index === 'number'
            ? String(item.clothing_category_index)
            : item.clothing_category_index;
        
        const processedColors: ColorInfo[] = (item.colors || []).map((c: any) => {
            const colorInfo: ColorInfo = { percentage: 0 };
            // console.log(`\n--- Color processing for ${item.class_name || item.className || 'item'} ---`);
            
            // Handle array format [name, percentage, rgb]
            if (Array.isArray(c) && c.length >= 3 && Array.isArray(c[2])) {
                colorInfo.rgb = c[2] as [number, number, number];
                colorInfo.percentage = c[1];
                // console.log(`Array format color: [${c[0]}]`);
                // console.log(`RGB: ${colorInfo.rgb.join(', ')}`);
                // console.log(`Percentage: ${colorInfo.percentage}%`);
                return colorInfo;
            }
            
            // Handle object format with rgb and percentage
            if (typeof c === 'object' && c !== null) {
                if (c.rgb && Array.isArray(c.rgb) && c.rgb.length === 3) {
                    colorInfo.rgb = c.rgb as [number, number, number];
                    // console.log(`RGB: ${colorInfo.rgb.join(', ')}`);
                }
                if (c.hsv && Array.isArray(c.hsv) && c.hsv.length === 3) {
                    colorInfo.hsv = c.hsv as [number, number, number];
                    // console.log(`HSV (existing): ${colorInfo.hsv.join(', ')}`);
                }
                if (c.percentage !== undefined) {
                    colorInfo.percentage = c.percentage;
                    // console.log(`Percentage: ${colorInfo.percentage}%`);
                }
                // Only return if it has enough info for the recommender
                if ((colorInfo.rgb || colorInfo.hsv) && colorInfo.percentage !== undefined) {
                    return colorInfo;
                }
            }
            
            // console.log(`Skipping invalid color format`);
            return null;
        }).filter((c): c is ColorInfo => c !== null); // Filter out nulls and assert type

        // console.log(`Processed ${processedColors.length} valid colors for ${item.class_name || item.className || 'item'}`);

        // Process PCA values - convert principalComponents to pca_values if needed
        let pcaValues: Record<string, number> = {};
        
        // First check if item already has pca_values
        if (item.pca_values && Object.keys(item.pca_values).length > 0) {
            pcaValues = item.pca_values;
        } 
        // Then try to convert principalComponents to pca_values
        else if (item.principalComponents && item.principalComponents.length > 0) {
            item.principalComponents.forEach((value, index) => {
                pcaValues[`pca_${index}`] = value;
            });
            console.log(`Converted ${Object.keys(pcaValues).length} principalComponents to pca_values`);
        }

        return {
            item_present: 1.0,
            class_name: determineItemClass(item),
            clothing_category: item.clothing_category ?? item.category ?? "Unknown",
            clothing_category_index: categoryIndex ?? "-1",
            confidence: confidenceValue ?? 0.0,
            description: item.description ?? null,
            pca_values: pcaValues,
            colors: processedColors,
            filename: item.filename ?? null,
            segmented_image_filename: segmentedImageFilename,
            segmented_image: item.segmented_image || null,
            temperature_suitability: normalizeTemperature(item),
            seasons: item.seasons ?? [],
            item_id: item.item_id ?? item.id,
            uri: item.uri,
            processedUri: item.processedUri,
            id: item.id
        };
    }

    // Add helper to get item from tensor slot
    private getItemFromTensorSlot(tensorSlot: OutfitTensorSlot): WardrobeItem | null {
        if (!tensorSlot.item_present || !tensorSlot.class_name) return null;
        
        // Try to find by item_id first
        if (tensorSlot.item_id) {
            const item = this.wardrobeItems.find(item => 
                (item.item_id === tensorSlot.item_id) || (item.id === tensorSlot.item_id)
            );
            if (item) return item;
        }
        
        // Try to find by id (from WardrobeScreen)
        if (tensorSlot.id) {
            const item = this.wardrobeItems.find(item => item.id === tensorSlot.id);
            if (item) return item;
        }
        
        // Fall back to filename matching
        if (tensorSlot.filename) {
            const item = this.wardrobeItems.find(item => item.filename === tensorSlot.filename);
            if (item) return item;
        }
        
        // Last resort: URI matching (from WardrobeScreen)
        if (tensorSlot.uri) {
            const item = this.wardrobeItems.find(item => item.uri === tensorSlot.uri);
            if (item) return item;
        }
        
        return null;
    }

    // Convert tensor to outfit
    private tensorToOutfit(tensor: OutfitTensor): Outfit | null {
        if (!tensor || tensor.length !== 3) return null;
        
        const outfit: Outfit = {
            outer: null, base: null, bottom: null
        };
        
        for (let i = 0; i < 3; i++) {
            if (tensor[i].item_present === 1.0) {
                outfit[SLOT_ORDER[i] as keyof Outfit] = this.getItemFromTensorSlot(tensor[i]);
            }
        }
        
        return outfit;
    }

    // Modify getRandomItem for weighted selection
    private getRandomItem(
        categoryList: string[], 
        targetTemperature: number = 2,
        isOuterLayer: boolean = false 
    ): WardrobeItem | null {
        const availableItems: WardrobeItem[] = [];
        const weights: number[] = [];
        
        for (const className of categoryList) {
            const items = this.itemsByClass[className] || [];
            for (const item of items) {
                availableItems.push(item);
                
                // Calculate weight based on temperature match only
                let weight = 0.0;
                
                // Temperature match - support both formats
                const normalizedTemp = normalizeTemperature(item);
                if (normalizedTemp.includes(targetTemperature)) {
                    weight += 2.0;
                } else if (normalizedTemp.length) {
                    // Find closest temperature
                    const closest = normalizedTemp.reduce((prev, curr) => 
                        Math.abs(curr - targetTemperature) < Math.abs(prev - targetTemperature) ? curr : prev
                    );
                    weight += 1.0 * (1 - Math.abs(closest - targetTemperature) / 4);
                }
                
                // Apply outer layer temperature bonus (make temperature match more important)
                if (isOuterLayer) {
                    weight *= 2.0;
                }
                
                weights.push(weight);
            }
        }
        
        if (availableItems.length === 0) return null;
        
        // Weighted random selection
        const totalWeight = weights.reduce((sum, w) => sum + w, 0);
        let randomValue = Math.random() * totalWeight;
        
        for (let i = 0; i < availableItems.length; i++) {
            randomValue -= weights[i];
            if (randomValue <= 0) {
                return availableItems[i];
            }
        }
        
        return availableItems[availableItems.length - 1]; // Fallback
    }

    private generateInitialOutfitTensor(temperature: number): OutfitTensor | null {
        // Determine if we need outer layer based on temperature
        const layers = this.determineLayersForTemperature(temperature);
        
        // Create empty tensor slots
        const tensor: OutfitTensor = Array(3).fill(null).map(() => ({
            item_present: 0.0,
            class_name: null,
            clothing_category: "Unknown",
            clothing_category_index: "-1",
            confidence: 0.0,
            description: null,
            pca_values: {},
            colors: [] as ColorInfo[],
            filename: null,
            segmented_image_filename: null,
            temperature_suitability: [],
            seasons: []
        }));
        
        // Always create a base+bottom outfit
        let base = this.getItemWithValidImage(CLOTHING_CATEGORIES["base"], temperature);
        if (!base) {
            console.error("Failed to find suitable base item with valid image");
            return null;
        }
        tensor[1] = this.itemToTensorSlot(base);
        
        // Always include bottom layer - filter based on temperature
        // Only use shorts for high temperatures (3-4), otherwise use trousers/skirts
        let bottomCategories: string[];
        if (temperature === 4) {
            // For hottest temperature (Hot), exclusively use shorts
            bottomCategories = ["shorts"];
            console.log("Using exclusively shorts for hottest temperature");
        } else if (temperature === 3) {
            // For warm temperature, include shorts and other bottoms
            bottomCategories = CLOTHING_CATEGORIES["bottom"];
            console.log("Using mixed bottoms (including shorts) for warm temperature");
        } else {
            // For moderate to cold temperatures, exclude shorts
            bottomCategories = CLOTHING_CATEGORIES["bottom"].filter(category => category !== "shorts");
            console.log("Excluding shorts for cooler temperatures");
        }
        
        let bottom = this.getItemWithValidImage(bottomCategories, temperature);
        if (!bottom) {
            console.error("Failed to find suitable bottom item with valid image");
            return null;
        }
        tensor[2] = this.itemToTensorSlot(bottom);
        
        // Only add outer layer if appropriate for temperature
        if (layers.use_outer) {
            const outer = this.getItemWithValidImage(CLOTHING_CATEGORIES["outer"], temperature, true);
            if (outer) {
                tensor[0] = this.itemToTensorSlot(outer);
            }
        }
        
        return tensor;
    }
    
    // Helper method to get an item that has a valid image path
    private getItemWithValidImage(
        categoryList: string[], 
        targetTemperature: number = 2,
        isOuterLayer: boolean = false
    ): WardrobeItem | null {
        let attempts = 0;
        
        // Add debugging for shorts selection
        const hasShorts = categoryList.includes('shorts');
        if (categoryList === CLOTHING_CATEGORIES["bottom"] || hasShorts) {
            console.log(`Bottoms selection for temperature ${targetTemperature} - shorts ${hasShorts ? 'allowed' : 'not allowed'}`);
        }
        
        while (attempts < 10) {
            attempts++;
            const item = this.getRandomItem(categoryList, targetTemperature, isOuterLayer);
            
            if (!item) {
                console.warn(`No items found in categories: ${categoryList.join(', ')}`);
                return null;
            }
            
            // Add debugging for shorts selection
            if (item.class_name === 'shorts' || item.className === 'shorts') {
                console.log(`Selected shorts for temperature ${targetTemperature}`);
            }
            
            // Check if item has any valid image path
            if (item.segmented_image || item.processedUri || item.uri) {
                return item; // Found an item with at least one valid image path
            }
            
            console.warn(`Item ${item.id || item.item_id || 'unknown'} has no valid image path. Trying another...`);
        }
        
        console.error(`Failed to find item with valid image after 10 attempts`);
        // Return the last item even without image, to avoid complete failure
        return this.getRandomItem(categoryList, targetTemperature, isOuterLayer);
    }

    // Updated to be async
    public async generateBestOutfitWithContext(
        numIterations: number = 50,
        temperature: number = 2
    ): Promise<{ bestOutfit: Outfit | null, tensor: OutfitTensor | null, score: number }> {
        console.log(`Generating outfit using iterative refinement (${numIterations} iterations)...`);
        
        // Generate initial outfit
        let bestTensor = this.generateInitialOutfitTensor(temperature);
        if (!bestTensor) {
            console.error("Failed to generate initial outfit tensor");
            return { bestOutfit: null, tensor: null, score: 0 };
        }
        
        // Score the initial tensor
        let bestScore = await this.scoreTensorAsync(bestTensor, temperature);
        console.log(`Initial outfit score: ${bestScore.toFixed(2)}`);
        
        // Get layers configuration based on temperature
        const layers = this.determineLayersForTemperature(temperature);
        
        // Create array of valid slot indices to modify
        const validSlots: number[] = [];
        
        // For outfits
        validSlots.push(1, 2); // Base and Bottom
        if (layers.use_outer) {
            validSlots.push(0); // Outer
        }
        
        // Iterative refinement
        for (let i = 0; i < numIterations; i++) {
            // Exit early if no valid slots to modify
            if (validSlots.length === 0) {
                console.log("No valid slots to modify based on temperature settings.");
                break;
            }
            
            // Select random slot from valid slots array
            const randomIndex = Math.floor(Math.random() * validSlots.length);
            const slotIndex = validSlots[randomIndex];
            const currentCategory = SLOT_ORDER[slotIndex];
            
            // Create a copy of the current best tensor
            const candidateTensor: OutfitTensor = JSON.parse(JSON.stringify(bestTensor));
            
            // Find alternative item for the selected slot
            let categoryItems = CLOTHING_CATEGORIES[currentCategory];
            
            // Apply temperature-based filtering for bottoms
            if (currentCategory === 'bottom') {
                if (temperature === 4) {
                    // For hottest temperature (Hot), exclusively use shorts
                    categoryItems = ["shorts"];
                } else {
                    // For moderate to cold temperatures, exclude shorts
                    categoryItems = categoryItems.filter(category => category !== "shorts");
                }
            }
            
            // Check if this is an outer layer item
            const isOuterLayer = currentCategory === 'outer';
            
            const alternativeItem = this.getItemWithValidImage(categoryItems, temperature, isOuterLayer);
            
            if (alternativeItem) {
                // Replace item in the candidate tensor
                candidateTensor[slotIndex] = this.itemToTensorSlot(alternativeItem);
                
                // Score the candidate tensor - now awaiting TFLite prediction
                const candidateScore = await this.scoreTensorAsync(
                    candidateTensor, temperature
                );
                
                // If better, update best tensor and score
                if (candidateScore > bestScore) {
                    bestScore = candidateScore;
                    bestTensor = candidateTensor;
                    console.log(`Iteration ${i+1}: Found better outfit (score: ${bestScore.toFixed(2)}) by changing ${currentCategory}`);
                }
            }
        }
        
        // Convert final tensor to outfit
        const bestOutfit = this.tensorToOutfit(bestTensor);
        
        return { bestOutfit, tensor: bestTensor, score: bestScore };
    }

    // New helper method to get async scores
    private async scoreTensorAsync(
        tensor: OutfitTensor, 
        temperature: number
    ): Promise<number> {
        // Add logging to show the tensor structure
        // console.log("--------- Tensor Structure Being Passed to Model ---------");
        // console.log(`Mode: ${this.mode}, Temperature: ${temperature}`);
        // console.log(JSON.stringify(tensor.map(slot => ({
        //     item_present: slot.item_present,
        //     class_name: slot.class_name,
        //     clothing_category: slot.clothing_category,
        //     clothing_category_index: slot.clothing_category_index,
        //     has_pca: slot.pca_values && Object.keys(slot.pca_values).length > 0 ? true : false,
        //     pca_keys_count: slot.pca_values ? Object.keys(slot.pca_values).length : 0,
        //     pca_keys: slot.pca_values ? Object.keys(slot.pca_values).slice(0, 5) : [], // Show first 5 keys
        //     pca_values_sample: slot.pca_values ? 
        //         Object.entries(slot.pca_values).slice(0, 3).map(([k, v]) => `${k}:${v.toFixed(4)}`) : 
        //         [],
        //     colors_count: slot.colors?.length || 0,
        //     colors_sample: slot.colors && slot.colors.length > 0 ? slot.colors.slice(0, 1) : [],
        //     temperature_suitability: slot.temperature_suitability,
        //     item_id: slot.item_id || slot.id
        // })), null, 2));
        // console.log("--------------------------------------------------------");
        
        // Check if we're using TFJSScorer
        if (this.outfitScorer instanceof TFJSScorer) {
            // Use direct async prediction
            return this.outfitScorer.predictAsync(tensor, temperature);
        } else {
            // Fall back to synchronous scoring for other scorers
            return this.outfitScorer.scoreOutfitTensor(tensor, temperature, 0);
        }
    }

    // Updated to simplify for men's outfits (no dress logic)
    private determineLayersForTemperature(temperature: number): LayerFlags {
        // Temperature: 0 (very cold) to 4 (hot)
        const layers: LayerFlags = {
            use_outer: false
        };

        // Determine if outer layer is needed based on temperature
        if (temperature <= 1) { // Very Cold / Cold (0-1)
            layers.use_outer = true;
        } else if (temperature === 2) { // Moderate
            layers.use_outer = Math.random() < 0.5; // 50% chance for outer in moderate weather
        } else { // Warm or Hot (3-4)
            layers.use_outer = false;
        }
        
        return layers;
    }

    // Updated for the new outfit structure
    public displayOutfitDetails(outfit: Outfit | null, tensor: OutfitTensor | null): void {
        if (!outfit) {
            console.log("No outfit to display.");
            return;
        }

        console.log("\n--- Chosen Outfit Details ---");
        let itemCount = 0;
        for (const [category, item] of Object.entries(outfit)) {
            if (item) {
                itemCount++;
                // Use either class_name or className, prioritizing class_name
                const itemClassName = item.class_name || item.className || 'Unknown';
                console.log(`- ${category.charAt(0).toUpperCase() + category.slice(1)}: ${itemClassName}`);
                
                // Handle file locations (either filename or uri from WardrobeScreen)
                if (item.filename || item.uri) {
                    console.log(`    File: ${item.filename || item.uri || 'N/A'}`);
                }
                
                if (item.segmented_image || item.processedUri) {
                    console.log(`    Segmented Img: ${item.segmented_image || item.processedUri || 'N/A'}`);
                }
                
                // Handle temperature (either array or single value)
                if (item.temperature_suitability || item.temperature !== undefined) {
                    const tempDisplay = item.temperature_suitability ? 
                        item.temperature_suitability.join(', ') : 
                        item.temperature;
                    console.log(`    Temperature Suitability: ${tempDisplay || 'N/A'} (0=very cold, 4=hot)`);
                }
                
                if (item.seasons) {
                    console.log(`    Seasons: ${item.seasons.join(', ') || 'N/A'}`);
                }
            }
        }
        if (itemCount === 0) {
             console.log("The chosen outfit is empty.");
        }
    }

    // Updated to be async
    public async getNextOutfit(
        temperature: number = 2
    ): Promise<{ bestOutfit: Outfit | null, tensor: OutfitTensor | null, score: number }> {
        if (this.mode === 'swipe' && this.precomputedOutfits.length > 0) {
            // Return the next precomputed outfit
            const outfitData = this.precomputedOutfits[this.currentPrecomputedIndex];
            this.currentPrecomputedIndex = (this.currentPrecomputedIndex + 1) % this.precomputedOutfits.length;
            
            // Calculate the score for this precomputed outfit
            const score = await this.scoreTensorAsync(outfitData.tensor, temperature);
            
            return { bestOutfit: outfitData.outfit, tensor: outfitData.tensor, score };
        } else {
            // Fall back to generation for generate mode or if no precomputed outfits
            return this.generateBestOutfitWithContext(50, temperature);
        }
    }

    // Add beam search implementation method
    public async generateWithBeam(
        numIterations: number = 10,
        temperature: number = 2,
        beamWidth: number = 3
    ): Promise<{ bestOutfit: Outfit | null; tensor: OutfitTensor | null; score: number }> {
        console.log(`Generating outfit using beam search (${numIterations} iterations, beam width ${beamWidth})...`);
        
        // 1) Start with a beam of size=1: the initial tensor
        const initial = this.generateInitialOutfitTensor(temperature);
        if (!initial) {
            console.error("Failed to generate initial outfit tensor");
            return { bestOutfit: null, tensor: null, score: 0 };
        }
        let beam: { tensor: OutfitTensor; score: number }[] = [
            { tensor: initial, score: await this.scoreTensorAsync(initial, temperature) }
        ];
        
        console.log(`Initial outfit score: ${beam[0].score.toFixed(2)}`);
        
        // Get layers configuration based on temperature
        const layers = this.determineLayersForTemperature(temperature);
        
        for (let iter = 0; iter < numIterations; iter++) {
            const allCandidates: { tensor: OutfitTensor; score: number }[] = [];
            
            // 2) Expand each tensor in the beam
            for (const { tensor: parentTensor } of beam) {
                // Create array of valid slot indices to modify
                const validSlots: number[] = [1, 2]; // Base and Bottom always valid
                if (layers.use_outer) {
                    validSlots.push(0); // Outer layer only if appropriate for temperature
                }
                
                for (const slotIdx of validSlots) {
                    const currentCategory = SLOT_ORDER[slotIdx];
                    let categoryItems = CLOTHING_CATEGORIES[currentCategory];
                    
                    // Apply temperature-based filtering for bottoms
                    if (currentCategory === 'bottom') {
                        if (temperature === 4) {
                            // For hottest temperature (Hot), exclusively use shorts
                            categoryItems = ["shorts"];
                        } else {
                            // For moderate to cold temperatures, exclude shorts
                            categoryItems = categoryItems.filter(category => category !== "shorts");
                        }
                    }
                    
                    // For each class in this category, try a random item
                    for (const className of categoryItems) {
                        // Get a random item of this class that's suitable for the temperature
                        // Check if this is an outer layer (slot 0)
                        const isOuterLayer = slotIdx === 0;
                        const alt = this.getItemWithValidImage([className], temperature, isOuterLayer);
                        if (!alt) continue;
                        
                        // Make a copy & swap in the replacement
                        const child = JSON.parse(JSON.stringify(parentTensor)) as OutfitTensor;
                        child[slotIdx] = this.itemToTensorSlot(alt);
                        
                        // Score it
                        const score = await this.scoreTensorAsync(child, temperature);
                        allCandidates.push({ tensor: child, score });
                    }
                }
            }
            
            if (allCandidates.length === 0) {
                console.log(`No valid candidates found at iteration ${iter+1}. Stopping early.`);
                break;
            }
            
            // 3) Pick the top-B scoring ones to form the next beam
            allCandidates.sort((a, b) => b.score - a.score);
            beam = allCandidates.slice(0, beamWidth);
            
            console.log(`After iter ${iter+1}, beam top score = ${beam[0].score.toFixed(2)}`);
        }
        
        // 4) At the end, beam[0] is the best
        const best = beam[0];
        return {
            bestOutfit: this.tensorToOutfit(best.tensor),
            tensor: best.tensor,
            score: best.score
        };
    }
}


async function generateOutfit(
    mode: OutfitGeneratorMode = 'generate',
    temperature: number = 2,
    numOutfits: number = NUM_OUTFITS_TO_GENERATE,
    precomputedOutfits: PrecomputedOutfit[] = [],
    wardrobeItems: WardrobeItem[] = wardrobeData,
    beamSearch: boolean = false,
    beamWidth: number = 3
): Promise<{ bestOutfit: Outfit | null, tensor: OutfitTensor | null, score: number }> {
    // console.log(`Initializing Outfit Generator in ${mode} mode...`);
    try {
        // Create scorer first
        const scorer = new TFJSScorer();
        
        // Wait for scorer initialization before proceeding
        console.log("Waiting for model initialization before generating outfits...");
        const modelReady = await scorer.waitForInitialization();
        console.log(`Model initialization ${modelReady ? 'successful' : 'failed, using fallback scoring'}`);
        
        // Initialize generator with initialized scorer
        const generator = new OutfitGenerator(
            wardrobeItems, 
            scorer,  
            mode, 
            precomputedOutfits
        );

        // console.log(`\nRequesting outfit with context:`);
        // console.log(`  Mode: ${mode}`);
        // console.log(`  Temperature: ${temperature} (0=very cold, 4=hot)`);
        // console.log(`  Search Method: ${beamSearch ? `Beam search (width=${beamWidth})` : 'Hill climbing'}`);
        
        let result;
        
        if (mode === 'generate') {
            // Use beam search if specified, otherwise use hill climbing
            result = beamSearch 
                ? await generator.generateWithBeam(numOutfits, temperature, beamWidth)
                : await generator.generateBestOutfitWithContext(numOutfits, temperature);
        } else {
            // For swipe mode, always use getNextOutfit
            result = await generator.getNextOutfit(temperature);
        }

        if (result.bestOutfit) {
            generator.displayOutfitDetails(result.bestOutfit, result.tensor);
            console.log(`\nOutfit ${mode === 'generate' ? 'generation' : 'selection'} complete.`);
            console.log(`Outfit score: ${result.score.toFixed(2)}`);
            return result;
        } else {
            console.log(`\nFailed to ${mode === 'generate' ? 'generate' : 'select'} a suitable outfit.`);
            return { bestOutfit: null, tensor: null, score: 0 };
        }
    } catch (error) {
        console.error("Error in outfit generation:", error);
        return { bestOutfit: null, tensor: null, score: 0 };
    }
}

export { generateOutfit };
