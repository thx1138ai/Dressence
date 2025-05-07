/**
 * This React Native screen implements the swiping UI for user feedback on AI-generated outfits:
 *
 * - Sets up TFJSRecommender and PreferenceService, loading models and updating stored preference confidences.
 * - Uses two overlapping animated cards (A & B) displaying recommended outfits, enabling swipe gestures for like/dislike.
 * - Manages card animations, pan responders, and regenerates new recommendations as cards are swiped.
 * - Records each swipe with outfit tensor, temperature, like/dislike, timestamp, and model confidence, stored via AsyncStorage.
 * - Triggers automatic model retraining after configurable swipe thresholds, updates training logs, and persists trained models.
 * - Provides a settings modal to view model status, copy detailed training logs, reset the model and clear all preferences.
 * - Displays live training progress, summary stats, and handles full model reset with error handling and UI feedback.
 */

import React, { useState, useRef, useEffect, useMemo } from 'react';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { 
  Platform, 
  Dimensions, 
  View, 
  Text, 
  Image, 
  Animated, 
  PanResponder, 
  SafeAreaView, 
  StyleSheet, 
  TouchableOpacity,
  ImageSourcePropType,
  ActivityIndicator,
  Alert,
  Modal,
  ToastAndroid,
  Clipboard,
  Easing,
  LogBox
} from 'react-native';

LogBox.ignoreLogs(['Clipboard has been extracted']);

import { TFJSRecommender } from './services/TFJSRecommender';
import { PreferenceService, StoredPreference, PreferenceStats } from './services/PreferenceService';
import { OutfitGenerator, Outfit, OutfitTensor, OutfitTensorSlot, WardrobeItem, generateOutfit } from './services/outfitGenerator';
import wardrobeData from './assets/precomputed_wardrobe.json';

import * as tf from '@tensorflow/tfjs';

import XIcon from './assets/icons/X.png';
import HeartIcon from './assets/icons/Heart.png';
import WardrobeIcon from './assets/icons/Wardrobe.png';
import MenuIcon from './assets/icons/Menu.png';

const SCREEN_WIDTH = Dimensions.get('window').width;
const SCREEN_HEIGHT = Dimensions.get('window').height;
const SWIPE_THRESHOLD = 0.25 * SCREEN_WIDTH;
const SWIPE_OUT_DURATION = 250;
const MIN_PREFERENCES_FOR_TRAINING = 20;

// Define slot order for consistency
const SLOT_ORDER = ['outer', 'base', 'bottom'] as const;
type SlotName = typeof SLOT_ORDER[number];

type Direction = 'left' | 'right';

type RootStackParamList = {
    Swipe: undefined;
    Wardrobe: undefined;
};

type SwipeScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Swipe'>;

type Props = {
    navigation: SwipeScreenNavigationProp;
};

type OutfitData = {
    outfit: Outfit | null;
    tensor: OutfitTensor | null;
    temperature: number;
    score?: number;
};

type EpochMetric = {
  epoch: number;
  train_loss: number;
  train_acc: number | null;
  val_loss: number | null;
  val_acc: number | null;
  timestamp: string;
};

type TrainingSummary = {
  totalSamples: number;
  likedSamples: number;
  dislikedSamples: number;
  averageConfidence?: number;
  finalTrainLoss?: number;
  finalTrainAcc?: number | null;
  lastTrainedAt?: string;
};

type SampleLog = {
  liked: boolean;
  temperature: number;
  timestamp: string;
  confidence?: number;
};

type DetailedSampleLog = SampleLog & {
    outfitDetails?: {
        items: Array<{
            category: string;
            className: string;
            imageUri?: string;
        }>;
        temperature: number;
    };
    swipeDirection: 'left' | 'right';
};

type TrainingLogs = {
    epochMetrics: EpochMetric[];
    sampleLogs: SampleLog[];
    detailedLogs: DetailedSampleLog[];
    summary?: TrainingSummary;
};

function SwipeScreen({ navigation }: Props) {
    const [outfitDataA, setOutfitDataA] = useState<OutfitData | null>(null);
    const [outfitDataB, setOutfitDataB] = useState<OutfitData | null>(null);
    const [activeCardId, setActiveCardId] = useState<'A' | 'B'>('A'); // A starts active

    const [isSettingsVisible, setIsSettingsVisible] = useState(false);
    const [isResettingModel, setIsResettingModel] = useState(false);

    const [isTraining, setIsTraining] = useState(false);
    const [trainingStatusMessage, setTrainingStatusMessage] = useState('');
    const [trainingStats, setTrainingStats] = useState<PreferenceStats | null>(null);

    const [trainingLogs, setTrainingLogs] = useState<TrainingLogs>({
        epochMetrics: [],
        sampleLogs: [],
        detailedLogs: []
    });

    // Animation values for Card A
    const positionA = useRef(new Animated.ValueXY()).current;
    const scaleA = useRef(new Animated.Value(1)).current;

    // Animation values for Card B
    const positionB = useRef(new Animated.ValueXY()).current;
    const scaleB = useRef(new Animated.Value(0.95)).current;

    // Replace swipe lock ref with state
    const [isSwiping, setIsSwiping] = useState(false);
    
    // Service Refs
    const tfjsRecommender = useRef<TFJSRecommender | null>(null);
    const preferenceService = useRef(new PreferenceService());

    // Animation value for training spinner
    const spinValue = useRef(new Animated.Value(0)).current;
    
    // Start spinning animation when training starts
    useEffect(() => {
        if (isTraining) {
            Animated.loop(
                Animated.timing(spinValue, {
                    toValue: 1,
                    duration: 3000,
                    easing: Easing.linear,
                    useNativeDriver: true
                })
            ).start();
        } else {
            spinValue.setValue(0);
        }
    }, [isTraining]);
    
    // Create spinning rotation from animated value
    const spin = spinValue.interpolate({
        inputRange: [0, 1],
        outputRange: ['0deg', '360deg']
    });

    // Initialise the Outfit Generator
    const outfitGenerator = useMemo(() => {
        console.log("Initializing Outfit Generator...");
        if (!Array.isArray(wardrobeData) || wardrobeData.length === 0) {
          console.error("Invalid or empty wardrobe data provided to OutfitGenerator!");
          return new OutfitGenerator([]);
        }
        return new OutfitGenerator(wardrobeData as WardrobeItem[]);
    }, []);

    useEffect(() => {
        tfjsRecommender.current = new TFJSRecommender();
        console.log('TFJS Recommender instance created.');
        tfjsRecommender.current.loadModel()
            .then(() => {
                console.log('TFJS Model loaded successfully for SwipeScreen.');
                
                // After model is loaded, update confidence scores for existing preferences
                const updateExistingPreferences = async () => {
                    if (tfjsRecommender.current?.isModelLoaded()) {
                        // Create a prediction function that can be passed to updateConfidenceScores
                        const predictor = async (tensor: OutfitTensor, temperature: number) => {
                            return await tfjsRecommender.current!.predict(tensor, temperature);
                        };
                        
                        // Update confidence scores for all preferences without them
                        const updatedCount = await preferenceService.current.updateConfidenceScores(predictor);
                        
                        if (updatedCount.updated > 0) {
                            console.log(`Updated confidence scores for ${updatedCount.updated} preferences out of ${updatedCount.total}`);
                            
                            // Refresh training logs with updated confidence scores
                            const prefs = await preferenceService.current.getPreferences();
                            
                            // Map preferences to sample logs format
                            const updatedLogs: SampleLog[] = prefs.map(p => ({
                                liked: p.liked,
                                temperature: p.temperature,
                                timestamp: p.timestamp,
                                confidence: p.confidence
                            }));
                            
                            // Update training logs
                            setTrainingLogs(prevLogs => ({
                                ...prevLogs,
                                sampleLogs: updatedLogs
                            }));
                            
                            // Update summary
                            setTimeout(updateTrainingLogsSummary, 0);
                        }
                    }
                };
                
                // Run the update
                updateExistingPreferences().catch(err => {
                    console.error('Error updating existing preferences:', err);
                });
            })
            .catch(err => console.error('Failed to load TFJS Model for SwipeScreen:', err));

        // Cleanup on unmount
        return () => {
            console.log('Closing TFJS Recommender on SwipeScreen unmount.');
            tfjsRecommender.current?.close();
        };
    }, []);

    // Load training stats on component mount
    useEffect(() => {
        const loadTrainingStats = async () => {
            try {
                const stats = await preferenceService.current.getTrainingStats();
                setTrainingStats(stats);
            } catch (error) {
                console.error("Error loading training stats:", error);
            }
        };
        
        loadTrainingStats();
    }, []);

    const generateRandomOutfitContext = () => {
        const temperature = Math.floor(Math.random() * 5);
        return { temperature };
    };

    const [isLoadingInitialOutfits, setIsLoadingInitialOutfits] = useState(true);
    const pulseAnim = useRef(new Animated.Value(0.6)).current;

    useEffect(() => {
        if (isLoadingInitialOutfits || !outfitDataA || !outfitDataB) {
            Animated.loop(
                Animated.sequence([
                    Animated.timing(pulseAnim, {
                        toValue: 1,
                        duration: 1000,
                        useNativeDriver: true,
                        easing: Easing.inOut(Easing.ease),
                    }),
                    Animated.timing(pulseAnim, {
                        toValue: 0.6,
                        duration: 1000,
                        useNativeDriver: true,
                        easing: Easing.inOut(Easing.ease),
                    })
                ])
            ).start();
        } else {
            // Stop animation when no longer loading
            pulseAnim.stopAnimation();
            pulseAnim.setValue(1);
        }
    }, [isLoadingInitialOutfits, outfitDataA, outfitDataB]);

    // Generate initial outfits on mount
    useEffect(() => {
        console.log("Generating initial outfits...");
        
        const generateInitialOutfits = async () => {
            setIsLoadingInitialOutfits(true);
            
            const contextA = generateRandomOutfitContext();
            const contextB = generateRandomOutfitContext();
            
            console.log("Initial outfit A parameters:", contextA);
            console.log("Initial outfit B parameters:", contextB);
            
            const initialA = await outfitGenerator.generateBestOutfitWithContext(1, contextA.temperature);
            const initialB = await outfitGenerator.generateBestOutfitWithContext(1, contextB.temperature);
            
            setOutfitDataA({
                outfit: initialA.bestOutfit,
                tensor: initialA.tensor,
                temperature: contextA.temperature,
                score: initialA.score
            });
            
            setOutfitDataB({
                outfit: initialB.bestOutfit,
                tensor: initialB.tensor,
                temperature: contextB.temperature,
                score: initialB.score
            });
            
            // Set loading to false when done
            setIsLoadingInitialOutfits(false);
            
            console.log("Initial outfits generated");
        };
        
        generateInitialOutfits();
    }, [outfitGenerator]);

    // Pan Responder for Card A
    const panResponderA = useMemo(() => PanResponder.create({
        onStartShouldSetPanResponder: () => activeCardId === 'A' && !isTraining && !isSwiping,
        onPanResponderMove: (_, gesture) => {
            if (activeCardId === 'A') {
                Animated.event(
                    [null, { dx: positionA.x, dy: positionA.y }],
                    { useNativeDriver: false }
                )(_, gesture);
            }
        },
        onPanResponderRelease: (_, gesture) => {
            if (activeCardId === 'A') {
                if (gesture.dx > SWIPE_THRESHOLD) {
                    forceSwipe('right');
                } else if (gesture.dx < -SWIPE_THRESHOLD) {
                    forceSwipe('left');
                } else {
                    resetPosition('A');
                }
            }
        },
    }), [activeCardId, isTraining, isSwiping]);

    // Pan Responder for Card B
    const panResponderB = useMemo(() => PanResponder.create({
        onStartShouldSetPanResponder: () => activeCardId === 'B' && !isTraining && !isSwiping,
        onPanResponderMove: (_, gesture) => {
            if (activeCardId === 'B') {
                Animated.event(
                    [null, { dx: positionB.x, dy: positionB.y }],
                    { useNativeDriver: false }
                )(_, gesture);
            }
        },
        onPanResponderRelease: (_, gesture) => {
            if (activeCardId === 'B') {
                if (gesture.dx > SWIPE_THRESHOLD) {
                    forceSwipe('right');
                } else if (gesture.dx < -SWIPE_THRESHOLD) {
                    forceSwipe('left');
                } else {
                    resetPosition('B');
                }
            }
        },
    }), [activeCardId, isTraining, isSwiping]);

    const resetPosition = (cardId: 'A' | 'B') => {
        const position = cardId === 'A' ? positionA : positionB;
        const RESET_DURATION = 200;

        Animated.timing(position, {
            toValue: { x: 0, y: 0 },
            duration: RESET_DURATION,
            useNativeDriver: false,
        }).start();
    };

    // Function to update summary statistics for training logs
    const updateTrainingLogsSummary = () => {
        const actualTotalSwipes = trainingStats?.totalSwipesEver || trainingLogs.sampleLogs.length;
        
        const summary: TrainingSummary = {
            totalSamples: actualTotalSwipes,
            likedSamples: trainingLogs.sampleLogs.filter(log => log.liked).length,
            dislikedSamples: trainingLogs.sampleLogs.filter(log => !log.liked).length,
            averageConfidence: trainingLogs.sampleLogs.length > 0 ?
                trainingLogs.sampleLogs
                    .filter(log => log.confidence !== undefined)
                    .reduce((sum, log) => sum + (log.confidence || 0), 0) / 
                    (trainingLogs.sampleLogs.filter(log => log.confidence !== undefined).length || 1)
                : undefined,
            finalTrainLoss: trainingLogs.epochMetrics.length > 0 ? 
                trainingLogs.epochMetrics[trainingLogs.epochMetrics.length - 1].train_loss : undefined,
            finalTrainAcc: trainingLogs.epochMetrics.length > 0 ? 
                trainingLogs.epochMetrics[trainingLogs.epochMetrics.length - 1].train_acc : undefined,
            lastTrainedAt: trainingLogs.epochMetrics.length > 0 ?
                trainingLogs.epochMetrics[trainingLogs.epochMetrics.length - 1].timestamp : undefined
        };
        
        setTrainingLogs(prevLogs => ({
            ...prevLogs,
            summary,
            detailedLogs: prevLogs.detailedLogs
        }));
    };

    const performAutomaticTraining = async () => {
        if (isTraining) return;
        if (!tfjsRecommender.current?.isModelLoaded()) {
            console.log("Model not ready for training, will try again later");
            return;
        }

        setIsTraining(true);
        setTrainingStatusMessage('Learning your outfit preferences...');

        try {
            const prefs = await preferenceService.current.getPreferences();
            console.log(`Starting automatic training with ${prefs.length} preferences`);

            const validPrefs = prefs.filter(p => 
                p.tensor && 
                Array.isArray(p.tensor) && 
                p.tensor.length > 0 && 
                typeof p.liked !== 'undefined' &&
                typeof p.temperature !== 'undefined'
            );

            if (validPrefs.length < MIN_PREFERENCES_FOR_TRAINING) {
                console.log(`Not enough valid preferences for training: ${validPrefs.length}/${MIN_PREFERENCES_FOR_TRAINING}`);
                setIsTraining(false);
                return;
            }

            // Create arrays for train()
            const outfitTensors = validPrefs.map(p => p.tensor!);
            const labels = validPrefs.map(p => p.liked ? 1 : 0);
            const temps = validPrefs.map(p => p.temperature);

            setTrainingStatusMessage(`Training with ${validPrefs.length} samples...`);

            // Prepare logs for this training session
            const newSampleLogs: SampleLog[] = validPrefs.map((p) => ({
                liked: p.liked,
                temperature: p.temperature,
                timestamp: p.timestamp,
                confidence: p.confidence
            }));

            // Kick off training
            const history = await tfjsRecommender.current!.train(
                outfitTensors,
                labels,
                temps,
                {
                    epochs: 10,
                    batchSize: 16,
                    validationSplit: 0.1,
                    classWeights: { 0: 1, 1: 1 }
                }
            );

            const epochMetrics: EpochMetric[] = [];
            
            if (history.history && history.history.loss) {
                const numEpochs = history.history.loss.length;
                
                for (let i = 0; i < numEpochs; i++) {
                    const metric: EpochMetric = {
                        epoch: i,
                        train_loss: typeof history.history.loss[i] === 'number' 
                            ? history.history.loss[i] as number
                            : (history.history.loss[i] as tf.Tensor).dataSync()[0],
                        train_acc: history.history.acc && history.history.acc[i] 
                            ? (typeof history.history.acc[i] === 'number'
                                ? history.history.acc[i] as number
                                : (history.history.acc[i] as tf.Tensor).dataSync()[0])
                            : null,
                        val_loss: history.history.val_loss && history.history.val_loss[i]
                            ? (typeof history.history.val_loss[i] === 'number'
                                ? history.history.val_loss[i] as number
                                : (history.history.val_loss[i] as tf.Tensor).dataSync()[0])
                            : null,
                        val_acc: history.history.val_acc && history.history.val_acc[i]
                            ? (typeof history.history.val_acc[i] === 'number'
                                ? history.history.val_acc[i] as number
                                : (history.history.val_acc[i] as tf.Tensor).dataSync()[0])
                            : null,
                        timestamp: new Date().toISOString()
                    };
                    epochMetrics.push(metric);
                }
            }

            setTrainingLogs(prevLogs => ({
                epochMetrics: [...prevLogs.epochMetrics, ...epochMetrics],
                sampleLogs: newSampleLogs,
                detailedLogs: prevLogs.detailedLogs
            }));
            
            setTimeout(updateTrainingLogsSummary, 0);

            await tfjsRecommender.current!.saveModel();
            
            await preferenceService.current.markTrained();
            
            const updatedStats = await preferenceService.current.getTrainingStats();
            setTrainingStats(updatedStats);

            console.log(`Training complete! Final loss: ${
                history.history && history.history.loss && history.history.loss.length > 0 ? 
                (typeof history.history.loss[history.history.loss.length - 1] === 'number' 
                    ? (history.history.loss[history.history.loss.length - 1] as number).toFixed(4)
                    : ((history.history.loss[history.history.loss.length - 1] as tf.Tensor).dataSync()[0]).toFixed(4)
                ) : 'N/A'
            }`);
            setTrainingStatusMessage('Training complete!');
            
            setTimeout(() => setTrainingStatusMessage(''), 3000);
        } catch (err: any) {
            console.error("Automatic training error:", err);
            setTrainingStatusMessage('Training failed, will try again later');
            setTimeout(() => setTrainingStatusMessage(''), 3000);
        } finally {
            setIsTraining(false);
        }
    };

    const forceSwipe = (direction: Direction) => {
        // if we're already in the middle of a swipe, bail out immediately
        if (isTraining || isSwiping) return;
        setIsSwiping(true);

        const isSwipingA = activeCardId === 'A';
        const outgoingPosition = isSwipingA ? positionA : positionB;
        const incomingPosition = isSwipingA ? positionB : positionA;
        const incomingScale = isSwipingA ? scaleB : scaleA;
        const outgoingData = isSwipingA ? outfitDataA : outfitDataB;

        const swipeTargetX = direction === 'right' ? SCREEN_WIDTH * 1.1 : -SCREEN_WIDTH * 1.1;

        console.log(`Swiping ${activeCardId} ${direction}. Incoming: ${isSwipingA ? 'B' : 'A'}`);

        Animated.parallel([
            // Animate outgoing card position
            Animated.timing(outgoingPosition, {
                toValue: { x: swipeTargetX, y: 0 },
                duration: SWIPE_OUT_DURATION,
                useNativeDriver: false,
            }),
            // Animate incoming card position to center
            Animated.timing(incomingPosition, {
                toValue: { x: 0, y: 0 },
                duration: SWIPE_OUT_DURATION,
                useNativeDriver: false,
            }),
            // Animate incoming card scale to 1
            Animated.timing(incomingScale, {
                toValue: 1,
                duration: SWIPE_OUT_DURATION,
                useNativeDriver: false,
            }),
        ]).start(async () => {
            console.log(`Swipe complete. New active: ${isSwipingA ? 'B' : 'A'}`);
            
            // Reset the visual properties of the outgoing card (now the background card)
            const outgoingCardId = activeCardId;
            const newActiveCardId = isSwipingA ? 'B' : 'A';
            const outgoingPositionAnim = isSwipingA ? positionA : positionB;
            const outgoingScaleAnim = isSwipingA ? scaleA : scaleB;
            
            // Set new active card
            setActiveCardId(newActiveCardId);
            
            // Reset animation values
            outgoingPositionAnim.setValue({ x: 0, y: 0 });
            outgoingScaleAnim.setValue(0.95);
            
            // Ensure the new active card's values are definitively set
            incomingPosition.setValue({ x: 0, y: 0 });
            incomingScale.setValue(1);
            
            // Release the swipe lock BEFORE starting async work
            setIsSwiping(false);
            
            console.log(`Reset visuals for ${outgoingCardId}`);

            // Save Preference via Service and check if we should train
            const liked = direction === 'right';
            if (outgoingData?.tensor) {
                // Get model prediction for this outfit (confidence score) 
                let confidenceScore: number | undefined = undefined;
                if (tfjsRecommender.current?.isModelLoaded()) {
                    try {
                        // Predict with the model to get confidence score
                        const prediction = await tfjsRecommender.current.predict(
                            outgoingData.tensor,
                            outgoingData.temperature
                        );
                        confidenceScore = prediction;
                        console.log(`Model prediction for swiped outfit: ${confidenceScore?.toFixed(4)}`);
                    } catch (predError) {
                        console.warn("Could not get model prediction:", predError);
                    }
                }
                
                // Create preference data with confidence score
                const preferenceData: StoredPreference = {
                    tensor: outgoingData.tensor,
                    temperature: outgoingData.temperature,
                    liked: liked,
                    timestamp: new Date().toISOString(),
                    confidence: confidenceScore
                };
                
                try {
                    // Add to our training logs
                    const newLog: SampleLog = {
                        liked,
                        temperature: outgoingData.temperature,
                        timestamp: new Date().toISOString(),
                        confidence: confidenceScore
                    };
                    
                    const detailedLog: DetailedSampleLog = {
                        ...newLog,
                        swipeDirection: direction as 'left' | 'right',
                        outfitDetails: {
                            items: outgoingData.tensor.map((slot, index) => {
                                // Get the category name based on index
                                const category = SLOT_ORDER[index] || 'unknown';
                                return {
                                    category: category,
                                    className: slot.class_name || 'unknown',
                                    imageUri: slot.segmented_image_filename || slot.processedUri || slot.uri || slot.segmented_image || undefined
                                };
                            }).filter(item => item.className !== 'unknown'), // Filter out empty slots
                            temperature: outgoingData.temperature
                        }
                    };
                    
                    setTrainingLogs(prevLogs => ({
                        ...prevLogs,
                        sampleLogs: [...prevLogs.sampleLogs, newLog],
                        detailedLogs: [...prevLogs.detailedLogs, detailedLog]
                    }));
                    
                    // Update summary after adding new log
                    setTimeout(updateTrainingLogsSummary, 0);
                    
                    // Updated addPreference now returns training information
                    const { shouldTrain, totalCount } = await preferenceService.current.addPreference(preferenceData);
                    
                    // IMPORTANT: Update local training stats state immediately to ensure counts display correctly
                    setTrainingStats(prevStats => {
                        // If prevStats is null, create initial stats object
                        const currentStats = prevStats || {
                            totalSwipesEver: 0,
                            swipesSinceLastTrain: 0,
                            lastTrainedAt: null
                        };
                        
                        return {
                            ...currentStats,
                            // Use the totalCount from service, but ensure it's at least one higher than before
                            totalSwipesEver: Math.max(totalCount, currentStats.totalSwipesEver + 1),
                            // Also increment swipes since last train
                            swipesSinceLastTrain: currentStats.swipesSinceLastTrain + 1
                        };
                    });
                    
                    // Now get the official stats from AsyncStorage to ensure we're fully in sync
                    const updatedStats = await preferenceService.current.getTrainingStats();
                    
                    // Only update the state if the new stats are at least as high as our local count
                    // This prevents the UI from showing a lower count if there's an AsyncStorage delay
                    setTrainingStats(prevStats => {
                        // If for some reason prevStats became null, use updatedStats
                        if (!prevStats) return updatedStats;
                        
                        return {
                            ...updatedStats,
                            totalSwipesEver: Math.max(updatedStats.totalSwipesEver, prevStats.totalSwipesEver),
                            swipesSinceLastTrain: Math.max(updatedStats.swipesSinceLastTrain, prevStats.swipesSinceLastTrain)
                        };
                    });
                    
                    console.log(`Preference saved. Total: ${totalCount}, Should train: ${shouldTrain}`);
                    
                    if (shouldTrain) {
                        // Show training status to user
                        if (updatedStats.lastTrainedAt === null) {
                            setTrainingStatusMessage(`First training session with ${totalCount} swipes`);
                        } else {
                            setTrainingStatusMessage(`Training with ${updatedStats.swipesSinceLastTrain} new swipes`);
                        }
                        
                        setTimeout(() => {
                            performAutomaticTraining();
                        }, 500);
                    }
                } catch (error) {
                    console.error("Error saving preference via service:", error);
                }
            }

            // Generate new outfit data for the card that just swiped out
            const randomContext = generateRandomOutfitContext();
            console.log(`Generating outfit with parameters:`, randomContext);
            
            // Await the async result
            const newNextData = await outfitGenerator.generateBestOutfitWithContext(
                5,
                randomContext.temperature
            );
            const setOutgoingData = outgoingCardId === 'A' ? setOutfitDataA : setOutfitDataB;
            setOutgoingData({ 
                outfit: newNextData.bestOutfit, 
                tensor: newNextData.tensor,
                temperature: randomContext.temperature,
                score: newNextData.score
            });
            console.log(`Generated new outfit for ${outgoingCardId} with score: ${newNextData.score?.toFixed(2) || 'N/A'}`);
        });
    };

    // Helper method to determine if outfit has outer layer
    const hasOuterLayer = (tensor: OutfitTensor): boolean => {
        return tensor.some((slot, idx) => 
            slot.item_present === 1.0 && idx === 0 // idx 0 is 'outer' in SLOT_ORDER
        );
    };

    // Card Styles
    const getAnimatedCardStyle = (position: Animated.ValueXY, scale: Animated.Value) => {
        const rotate = position.x.interpolate({
            inputRange: [-SCREEN_WIDTH / 2, 0, SCREEN_WIDTH / 2],
            outputRange: ['-10deg', '0deg', '10deg'],
            extrapolate: 'clamp',
        });
        
        // Calculate optimal card size based on screen dimensions
        const cardWidth = Math.min(SCREEN_WIDTH * 0.9, 375);
        const cardHeight = Math.min(
            // For small screens, use a more compact height
            SCREEN_HEIGHT < 700 ? SCREEN_HEIGHT * 0.65 : SCREEN_HEIGHT * 0.7, 
            SCREEN_WIDTH * 1.35
        );
        
        return {
            position: 'absolute' as const,
            width: cardWidth,
            height: cardHeight,
            opacity: scale.interpolate({
                inputRange: [0.9, 1],
                outputRange: [0.8, 1],
                extrapolate: 'clamp'
            }),
            transform: [
                { translateX: position.x },
                { translateY: position.y },
                { rotate },
                { scale }
            ],
        };
    };

    // --- Asset Loading Helper ---
    const getImageSource = (filename: string | null | undefined): ImageSourcePropType | null => {
        if (!filename) {
            return null;
        }
        
        try {
            return { uri: filename };
        } catch (error) {
            console.warn(`Image asset not found: ${filename}`);
            return null;
        }
    };

    // Function to render a single outfit card
    const renderOutfitCard = (
        outfitData: OutfitData | null,
        animatedStyle: any
    ) => {
        if (!outfitData || !outfitData.tensor) {
            return (
                <View style={styles.cardStyle}>
                    {/* Base background color for the card */}
                    <View style={styles.cardBackground} />
                    
                    {/* Centered loading content */}
                    <View style={styles.loadingCardContent}>
                        <Animated.View 
                            style={[
                                styles.loadingIconContainer,
                                { opacity: pulseAnim }
                            ]}
                        >
                            <View style={styles.loadingCircle}>
                                <ActivityIndicator size="large" color="#007AFF" />
                            </View>
                        </Animated.View>
                        <Text style={styles.loadingCardTitle}>Preparing Outfit</Text>
                        <Text style={styles.loadingCardSubtitle}>Finding the perfect style for you...</Text>
                    </View>
                </View>
            );
        }

        // Get temperature labels
        const temperatureLabels = ['Cold', 'Cool', 'Moderate', 'Warm', 'Hot'];
        const temperatureLabel = temperatureLabels[outfitData.temperature];
        
        // Get appropriate emoji for temperature
        const getTempEmoji = (temp: number) => {
            const emojis = ['❄️', '🧊', '🌤️', '☀️', '🔥'];
            return emojis[temp];
        };
        
        // Get colour for temperature badge
        const getTempColor = (temp: number) => {
            const colors = ['#7cb9e8', '#98d7e4', '#e5d8bf', '#f5c65d', '#e57b5c'];
            return colors[temp];
        };

        const tensor = outfitData.tensor;
        const hasOuter = hasOuterLayer(tensor);

        // Layout styles for each slot
        const layoutStyles: Record<SlotName, any> = {
            outer: {
                position: 'absolute',
                top: '5%',
                left: '35%',
                width: '63%',
                height: '52.5%',
                zIndex: 4,
            },
            base: {
                position: 'absolute',
                top: hasOuter ? '10%' : '8%',
                left: hasOuter ? '5%' : '20%',
                width: '60%',
                height: '50%',
                zIndex: hasOuter ? 2 : 3,
            },
            bottom: {
                position: 'absolute',
                top: hasOuter ? '52%' : '50%',
                left: '27.5%',
                width: '45%',
                height: tensor[2].class_name === 'shorts' ? 
                       (hasOuter ? '32%' : '35%') : 
                       (hasOuter ? '45%' : '47%'),
                zIndex: 1,
            }
        };

        // Function to get shadow style based on slot
        const shadowStyleFor = (which: SlotName) => {
            if (which === 'outer') {
                return {
                    shadowColor: '#000',
                    shadowOffset: { width: 4, height: 3 },
                    shadowOpacity: 0.4,
                    shadowRadius: 5,
                    elevation: 6,
                };
            }
            if (which === 'base') {
                return {
                    shadowColor: '#000',
                    shadowOffset: { width: 2, height: 3 },
                    shadowOpacity: 0.3,
                    shadowRadius: 3,
                    elevation: 4,
                };
            }
            return {
                shadowColor: '#000',
                shadowOffset: { width: 2, height: 2 },
                shadowOpacity: 0.25,
                shadowRadius: 2,
                elevation: 3,
            };
        };

        return (
            <View style={styles.cardStyle}>
                {/* Base background color for the card */}
                <View style={styles.cardBackground} />

                {/* Map through the tensor slots and render images */}
                {tensor.map((slot, idx) => {
                    // Only render if item is present
                    if (slot.item_present !== 1.0) return null;

                    // 1) figure out which slot this is
                    const which: SlotName = SLOT_ORDER[idx];

                    // 2) Get image source with fallback mechanism
                    let imageSource = null;
                    
                    // Try all possible image paths
                    if (slot.segmented_image_filename) {
                        imageSource = getImageSource(slot.segmented_image_filename);
                    }
                    
                    if (!imageSource && slot.processedUri) {
                        imageSource = getImageSource(slot.processedUri);
                    }
                    
                    if (!imageSource && slot.uri) {
                        imageSource = getImageSource(slot.uri);
                    }
                    
                    if (!imageSource && slot.segmented_image) {
                        imageSource = getImageSource(slot.segmented_image);
                    }

                    // Only render if we have a valid image source
                    if (!imageSource) return null;

                    // 3) Get the appropriate layout style
                    const style = layoutStyles[which];

                    // 4) Calculate rotation based on slot
                    const rotation = idx === 0 ? 3 : idx === 1 ? -3 : 0;

                    return (
                        <Animated.Image
                            key={`${which}-${idx}`}
                            source={imageSource}
                            style={[
                                style,
                                shadowStyleFor(which),
                                { 
                                    transform: [{ rotate: `${rotation}deg` }],
                                    opacity: which === 'outer' ? 0.9 : 
                                            which === 'base' ? 0.95 : 1  // Varying opacity for layers
                                }
                            ]}
                            resizeMode="contain"
                        />
                    );
                })}

                {/* Context info badges */}
                <View style={styles.contextContainer}>
                    {/* Temperature Badge */}
                    <View style={[styles.contextBadge, { backgroundColor: getTempColor(outfitData.temperature) }]}>
                        <Text style={styles.contextEmoji}>{getTempEmoji(outfitData.temperature)}</Text>
                        <Text style={styles.contextText}>{temperatureLabel}</Text>
                    </View>
                    
                    {/* Score Badge */}
                    {/* {outfitData.score !== undefined && (
                        <View style={[styles.contextBadge, { backgroundColor: '#8bc34a' }]}>
                            <Text style={styles.contextEmoji}>⭐</Text>
                            <Text style={styles.contextText}>Score: {outfitData.score.toFixed(2)}</Text>
                        </View>
                    )} */}
                </View>
            </View>
        );
    };

    // Header with navigation icons
    const renderHeader = () => (
        <View style={styles.header}>
            <TouchableOpacity
                onPress={() => setIsSettingsVisible(true)}
                hitSlop={{top: 10, bottom: 10, left: 10, right: 10}}
                style={styles.headerButton}
            >
                <Image source={MenuIcon} style={styles.headerIcon} resizeMode="contain" />
            </TouchableOpacity>
            <TouchableOpacity 
                onPress={() => navigation.navigate('Wardrobe')}
                hitSlop={{top: 10, bottom: 10, left: 10, right: 10}}
                style={styles.headerButton}
            >
                <Image source={WardrobeIcon} style={styles.headerIcon} resizeMode="contain" />
            </TouchableOpacity>
        </View>
    );

    // Function to copy logs to clipboard - create a user-friendly summary
    const copyLogsToClipboard = async () => {
        try {
            // Make sure summary is updated
            updateTrainingLogsSummary();
            
            // Create a simplified log object
            const simplifiedLogs: any = {
                summary: trainingLogs.summary || {
                    totalSamples: trainingStats?.totalSwipesEver || 0,
                    likedSamples: trainingLogs.sampleLogs.filter(log => log.liked).length,
                    dislikedSamples: trainingLogs.sampleLogs.filter(log => !log.liked).length,
                    averageConfidence: trainingLogs.sampleLogs.length > 0 ? 
                        trainingLogs.sampleLogs
                            .filter(log => log.confidence !== undefined)
                            .reduce((sum, log) => sum + (log.confidence || 0), 0) / 
                            (trainingLogs.sampleLogs.filter(log => log.confidence !== undefined).length || 1) : 0,
                    finalTrainLoss: trainingLogs.epochMetrics.length > 0 ? 
                        trainingLogs.epochMetrics[trainingLogs.epochMetrics.length - 1].train_loss : null,
                    modelStatus: tfjsRecommender.current?.isUsingPersistedModel() ? "Trained Model" : "Initial Model"
                },
                trainingHistory: trainingLogs.epochMetrics.map(metric => ({
                    epoch: metric.epoch,
                    loss: metric.train_loss.toFixed(4),
                    accuracy: metric.train_acc !== null ? metric.train_acc.toFixed(4) : 'N/A',
                    val_loss: metric.val_loss !== null ? metric.val_loss.toFixed(4) : 'N/A',
                    val_accuracy: metric.val_acc !== null ? metric.val_acc.toFixed(4) : 'N/A',
                    timestamp: metric.timestamp
                })),
                swipeHistory: trainingLogs.detailedLogs.map(log => ({
                    timestamp: log.timestamp,
                    liked: log.liked,
                    swipeDirection: log.swipeDirection,
                    confidence: log.confidence?.toFixed(4) || 'N/A',
                    temperature: log.temperature,
                    outfitItems: log.outfitDetails?.items.map(item => ({
                        category: item.category,
                        className: item.className,
                        imageUri: item.imageUri
                    })) || []
                })),
                logStatus: trainingLogs.sampleLogs.length === 0 && trainingLogs.epochMetrics.length === 0 ? 
                    "No training data available yet. Swipe on more outfits to generate data." : 
                    `Contains data for ${trainingLogs.sampleLogs.length} swipes and ${trainingLogs.epochMetrics.length} training epochs.`
            };

            const logsString = JSON.stringify(simplifiedLogs, null, 2);
            
            try {
                await Clipboard.setString(logsString);
                console.log("Clipboard operation completed");
            } catch (clipboardError: any) {
                console.log("Clipboard operation failed:", clipboardError);
            }
            
            if (Platform.OS === 'android') {
                ToastAndroid.show('Training summary copied to clipboard', ToastAndroid.SHORT);
            } else {
                Alert.alert('Success', 'Training summary copied to clipboard');
            }
        } catch (error) {
            console.error('Error copying logs to clipboard:', error);
            Alert.alert('Error', 'Failed to copy logs to clipboard');
        }
    };

    // Settings Modal Component
    const renderSettingsModal = () => {
        // Get model status
        const isUsingTrainedModel = tfjsRecommender.current?.isUsingPersistedModel() || false;
        
        return (
            <Modal
                animationType="slide"
                transparent={true}
                visible={isSettingsVisible}
                onRequestClose={() => setIsSettingsVisible(false)}
            >
                <View style={styles.modalOverlay}>
                    <View style={styles.modalContent}>
                        <View style={styles.modalHeader}>
                            <Text style={styles.modalTitle}>Settings</Text>
                            <TouchableOpacity onPress={() => setIsSettingsVisible(false)}>
                                <Text style={styles.closeButton}>✕</Text>
                            </TouchableOpacity>
                        </View>

                        {/* Model Status Indicator */}
                        <View style={styles.modelStatusContainer}>
                            <Text style={styles.modelStatusLabel}>
                                Model Status:
                            </Text>
                            <View style={[
                                styles.modelStatusBadge, 
                                isUsingTrainedModel ? styles.modelStatusTrainedBadge : styles.modelStatusOriginalBadge
                            ]}>
                                <Text style={styles.modelStatusText}>
                                    {isUsingTrainedModel ? 'Using Trained Model' : 'Using Original Model'}
                                </Text>
                            </View>
                        </View>
                        
                        {/* Copy Logs Button */}
                        <TouchableOpacity 
                            style={[styles.copyLogsButton]}
                            onPress={copyLogsToClipboard}
                        >
                            <Text style={styles.copyLogsButtonText}>Copy Training Logs</Text>
                        </TouchableOpacity>
                        
                        {/* Reset Model Button */}
                        <TouchableOpacity 
                            style={[styles.resetModelButton, isResettingModel && styles.resetModelButtonDisabled]}
                            onPress={() => {
                                Alert.alert(
                                    "Reset Model",
                                    "This will reset the recommendation model and clear all your saved preferences. This cannot be undone.",
                                    [
                                        {
                                            text: "Cancel",
                                            style: "cancel"
                                        },
                                        { 
                                            text: "Reset", 
                                            style: "destructive",
                                            onPress: handleResetModel
                                        }
                                    ]
                                );
                            }}
                            disabled={isResettingModel}
                        >
                            {isResettingModel ? (
                                <ActivityIndicator color="#fff" size="small" />
                            ) : (
                                <Text style={styles.resetModelButtonText}>Reset Model</Text>
                            )}
                        </TouchableOpacity>
                    </View>
                </View>
            </Modal>
        );
    };

    // Render function to display training progress information
    const renderTrainingInfo = () => {
        if (!trainingStats) return null;
        
        // Get progress data
        let progressText = '';
        let progress = 0;
        let totalNeeded = 0;
        
        if (trainingStats.lastTrainedAt === null) {
            // First-time training, need 50 swipes
            totalNeeded = 50;
            progress = trainingStats.totalSwipesEver;
            progressText = `Initial training: ${progress}/${totalNeeded}`;
        } else {
            // Later training, need 25 more swipes
            totalNeeded = 25;
            progress = trainingStats.swipesSinceLastTrain;
            progressText = `Next training: ${progress}/${totalNeeded}`;
        }

        const progressPercent = Math.min(100, (progress / totalNeeded) * 100);
        
        return (
            <View style={styles.trainingInfoContainer}>
                <View style={styles.trainingProgressWrapper}>
                    <Text style={styles.trainingProgressText}>{progressText}</Text>
                    <View style={styles.progressBarContainer}>
                        <View 
                            style={[
                                styles.progressBar, 
                                { width: `${progressPercent}%` as any }
                            ]} 
                        />
                    </View>
                </View>
            </View>
        );
    };

    // Reset model function
    const handleResetModel = async () => {
        if (isResettingModel) return;
        
        setIsResettingModel(true);
        setTrainingStatusMessage('Resetting model, please wait...');
        
        // Set a timeout to prevent UI from being stuck if reset hangs
        const resetTimeout = setTimeout(() => {
            console.log("Reset operation timed out");
            setIsResettingModel(false);
            setTrainingStatusMessage('');
            setIsSettingsVisible(false);
            Alert.alert(
                "Reset Timeout", 
                "The reset operation took too long. Please restart the app and try again."
            );
        }, 15000); // 15 second timeout
        
        try {
            // Check if we have a recommender instance
            if (tfjsRecommender.current) {
                console.log("Performing complete model reset...");
                
                await tfjsRecommender.current.resetTensorMemory();
                
                // Reset preferences and stats (more thorough than just clearing)
                console.log("Resetting saved preferences and stats...");
                await preferenceService.current.resetPreferencesAndStats();
                
                // Reset training logs
                setTrainingLogs({
                    epochMetrics: [],
                    sampleLogs: [],
                    detailedLogs: []
                });
                
                // Update training stats after reset
                setTrainingStats({
                    totalSwipesEver: 0,
                    swipesSinceLastTrain: 0,
                    lastTrainedAt: null
                });
                
                console.log("Model reset complete");
                
                // Clear timeout since we completed successfully
                clearTimeout(resetTimeout);
                
                // Update UI
                setIsResettingModel(false);
                setTrainingStatusMessage('');
                setIsSettingsVisible(false);
                
                // Show success message
                Alert.alert(
                    "Restart your App. Reset Complete.", 
                    "Model has been reset successfully. Your preferences have been cleared and the original model will be used. Please restart the app to ensure the reset takes full effect and avoid potential errors."
                );
            } else {
                clearTimeout(resetTimeout);
                console.error("No recommender instance found");
                setIsResettingModel(false);
                setTrainingStatusMessage('');
                Alert.alert("Error", "Could not access the recommendation model. Please restart the app and try again.");
            }
        } catch (error) {
            clearTimeout(resetTimeout);
            console.error("Error resetting model:", error);
            setIsResettingModel(false);
            setTrainingStatusMessage('');
            setIsSettingsVisible(false);
            Alert.alert("Reset Failed", "There was a problem resetting the model. Please restart the app and try again.");
        }
    };

    // Main render function
    return (
        <SafeAreaView style={styles.container}>
            {renderHeader()}
            {renderSettingsModal()}

            {/* Enhanced Training Overlay */}
            {(isTraining || (isResettingModel && trainingStatusMessage !== '')) && (
                <View style={styles.trainingOverlay}>
                    <View style={styles.trainingModal}>
                        <Animated.View 
                            style={[
                                styles.spinnerContainer, 
                                { transform: [{ rotate: spin }] }
                            ]}
                        >
                            <View style={styles.spinnerRing}>
                                <View style={styles.spinnerCore} />
                            </View>
                        </Animated.View>
                        <Text style={styles.trainingTitle}>{isTraining ? "Learning" : "Resetting"}</Text>
                        <Text style={styles.trainingStatusText}>
                            {trainingStatusMessage || (isTraining ? 'We are learning your outfit preferences...' : 'Resetting the recommendation model...')}
                        </Text>
                    </View>
                </View>
            )}
            
            {/* Training Info Indicator - always visible when not training */}
            {!isTraining && trainingStatusMessage === '' && renderTrainingInfo()}

            <View style={styles.contentArea}>
                <View style={styles.cardStackWrapper}>
                    {/* Render loading or error state if outfits aren't ready */}
                    {!outfitDataA || !outfitDataB ? (
                        <View style={styles.cardStyle}>
                            {/* Base background color for the card */}
                            <View style={styles.cardBackground} />
                            
                            {/* Centered loading content */}
                            <View style={styles.loadingCardContent}>
                                <Animated.View 
                                    style={[
                                        styles.loadingIconContainer,
                                        { opacity: pulseAnim }
                                    ]}
                                >
                                    <View style={styles.loadingCircle}>
                                        <ActivityIndicator size="large" color="#007AFF" />
                                    </View>
                                </Animated.View>
                                <Text style={styles.loadingCardTitle}>Starting Up</Text>
                                <Text style={styles.loadingCardSubtitle}>Preparing your fashion recommendations...</Text>
                            </View>
                        </View>
                    ) : (
                        <>
                            {/* Card B - Rendered first so it's behind A by default */}
                            <Animated.View
                                style={[
                                    getAnimatedCardStyle(positionB, scaleB),
                                    { zIndex: activeCardId === 'B' ? 1 : 0 } // Bring B to front if active
                                ]}
                                {...(activeCardId === 'B' && !isTraining ? panResponderB.panHandlers : {})}
                            >
                                {renderOutfitCard(outfitDataB, {})}
                            </Animated.View>

                            {/* Card A - Rendered second so it's on top by default */}
                            <Animated.View
                                style={[
                                    getAnimatedCardStyle(positionA, scaleA),
                                    { zIndex: activeCardId === 'A' ? 1 : 0 } // Bring A to front if active
                                ]}
                                {...(activeCardId === 'A' && !isTraining ? panResponderA.panHandlers : {})}
                            >
                                {renderOutfitCard(outfitDataA, {})}
                            </Animated.View>
                        </>
                    )}
                </View>
                
                {/* Buttons - now just Like/Dislike, no Train button */}
                {outfitDataA && outfitDataB && (
                    <View style={styles.buttonContainer}>
                        {/* Dislike Button */}
                        <TouchableOpacity 
                            style={styles.swipeButton}
                            onPress={() => forceSwipe('left')}
                            disabled={isTraining || isSwiping}
                        >
                            <Image source={XIcon} style={styles.iconStyle} resizeMode="contain" />
                            <Text style={styles.buttonText} allowFontScaling={false}>Not a Fan</Text>
                        </TouchableOpacity>

                        {/* Like Button */}
                        <TouchableOpacity 
                            style={styles.swipeButton}
                            onPress={() => forceSwipe('right')}
                            disabled={isTraining || isSwiping}
                        >
                            <Image source={HeartIcon} style={styles.iconStyle} resizeMode="contain" />
                            <Text style={styles.buttonText} allowFontScaling={false}>I'd wear that</Text>
                        </TouchableOpacity>
                    </View>
                )}
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
        paddingBottom: Platform.OS === 'ios' ? 20 : 0,
    },
    contentArea: {
        flex: 1,
        position: 'relative',
        justifyContent: 'center',
        paddingHorizontal: 5,
        paddingTop: 0,
        paddingBottom: 0,
    },
    cardStackWrapper: {
        flex: 0,
        height: Math.min(SCREEN_HEIGHT * 0.72, SCREEN_HEIGHT - 190),
        width: '100%',
        alignItems: 'center',
        justifyContent: 'center',
        marginTop: -70,
        marginBottom: 0,
    },
    cardStyle: {
        width: '100%',
        height: '100%',
        borderRadius: 24,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.15,
        shadowRadius: 8,
        elevation: 5,
        position: 'relative',
        backgroundColor: 'transparent',
        overflow: 'hidden',
        borderWidth: 1,
        borderColor: '#dcdcdc',
    },
    cardBackground: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: '#F4F4F4',
        borderRadius: 24,
    },
    placeholderCard: {
        backgroundColor: '#e0e0e0',
        alignItems: 'center',
        justifyContent: 'center',
    },
    placeholderText: {
        fontSize: 16,
        color: '#333',
        marginBottom: 10,
    },
    loadingContainer: {
        width: SCREEN_WIDTH * 0.9,
        height: SCREEN_WIDTH * 1.35,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#e0e0e0',
        borderRadius: 24,
    },
    loadingText: {
        fontSize: 16,
        marginBottom: 16,
        color: '#333',
    },
    buttonContainer: {
        position: 'absolute',
        bottom: Platform.OS === 'ios' ? 30 : 20,
        left: 0,
        right: 0,
        flexDirection: 'row',
        justifyContent: 'space-around',
        alignItems: 'center',
        paddingHorizontal: '5%',
    },
    swipeButton: {
        backgroundColor: 'white',
        paddingVertical: Math.min(15, SCREEN_WIDTH * 0.03),
        paddingHorizontal: Math.min(20, SCREEN_WIDTH * 0.04),
        borderRadius: 35,
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
        elevation: 4,
        width: SCREEN_WIDTH * 0.42,
        maxWidth: 160,
    },
    iconStyle: {
        width: Math.min(30, SCREEN_WIDTH * 0.06),
        height: Math.min(30, SCREEN_WIDTH * 0.06),
        marginRight: 8,
    },
    buttonText: {
        fontSize: Math.min(16, SCREEN_WIDTH * 0.038),
        fontWeight: '500',
        color: '#333',
        textAlign: 'center',
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 24,
        paddingTop: Platform.OS === 'ios' ? 0 : 16,
        paddingBottom: 16,
        width: '100%',
        backgroundColor: '#fff',
        zIndex: 10,
    },
    headerIcon: {
        width: Math.min(34, SCREEN_WIDTH * 0.09),
        height: Math.min(34, SCREEN_WIDTH * 0.09),
    },
    headerButton: {
        padding: 8,
        zIndex: 10,
    },
    contextContainer: {
        position: 'absolute',
        top: 20,
        right: 20,
        flexDirection: 'column',
        alignItems: 'flex-end',
        zIndex: 999,
    },
    contextBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 12,
        paddingVertical: 8,
        marginBottom: 10,
        borderRadius: 16,
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 3,
        elevation: 3,
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.6)',
    },
    contextEmoji: {
        fontSize: 16,
        marginRight: 6,
    },
    contextText: {
        fontSize: 13,
        fontWeight: '600',
        color: '#333',
    },
    trainingOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 1000,
    },
    trainingModal: {
        backgroundColor: 'white',
        borderRadius: 20,
        width: '80%',
        paddingVertical: 30,
        paddingHorizontal: 20,
        alignItems: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 5 },
        shadowOpacity: 0.5,
        shadowRadius: 10,
        elevation: 10,
    },
    spinnerContainer: {
        width: 120,
        height: 120,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 20,
    },
    spinnerRing: {
        width: 100,
        height: 100,
        borderRadius: 50,
        borderWidth: 8,
        borderColor: '#007AFF',
        borderTopColor: '#E0E0E0',
        justifyContent: 'center',
        alignItems: 'center',
    },
    spinnerCore: {
        width: 20,
        height: 20,
        borderRadius: 10,
        backgroundColor: '#007AFF',
    },
    trainingTitle: {
        fontSize: 22,
        fontWeight: 'bold',
        marginBottom: 10,
        color: '#333',
    },
    trainingStatusText: {
        fontSize: 16,
        color: '#666',
        textAlign: 'center',
        paddingHorizontal: 10,
    },
    trainingInfoContainer: {
        position: 'absolute',
        top: Platform.OS === 'ios' ? 60 : 70,
        left: 0,
        right: 0,
        paddingVertical: 8,
        backgroundColor: 'rgba(255, 255, 255, 0.0)',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 10,
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(255, 255, 255, 0.0)',
        pointerEvents: 'none',
    },
    trainingProgressWrapper: {
        width: '55%',
        alignItems: 'center',
    },
    trainingProgressText: {
        fontSize: 14,
        color: '#333',
        fontWeight: '600',
        marginBottom: 4,
    },
    progressBarContainer: {
        height: 6,
        width: '100%',
        backgroundColor: '#e0e0e0',
        borderRadius: 3,
        overflow: 'hidden',
    },
    progressBar: {
        height: '100%',
        backgroundColor: '#4CAF50',
        borderRadius: 3,
    },
    copyLogsButton: {
        backgroundColor: '#4CAF50',
        paddingVertical: 12,
        borderRadius: 8,
        alignItems: 'center',
        marginTop: 10,
        marginBottom: 10,
    },
    copyLogsButtonText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: 'bold',
    },
    modalOverlay: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
    },
    modalContent: {
        width: '85%',
        backgroundColor: 'white',
        borderRadius: 20,
        padding: 20,
        shadowColor: '#000',
        shadowOffset: {
            width: 0,
            height: 2
        },
        shadowOpacity: 0.25,
        shadowRadius: 3.84,
        elevation: 5
    },
    modalHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 20,
    },
    modalTitle: {
        fontSize: 24,
        fontWeight: 'bold',
    },
    closeButton: {
        fontSize: 24,
        fontWeight: 'bold',
    },
    settingHeader: {
        fontSize: 18,
        fontWeight: 'bold',
        marginBottom: 10,
        marginTop: 5,
    },
    preferenceBubbles: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent: 'flex-start',
        marginBottom: 20,
    },
    preferenceBubble: {
        paddingHorizontal: 12,
        paddingVertical: 8,
        backgroundColor: '#f0f0f0',
        borderRadius: 20,
        marginRight: 8,
        marginBottom: 8,
    },
    selectedBubble: {
        backgroundColor: '#007AFF',
    },
    bubbleText: {
        color: '#333',
        fontSize: 14,
    },
    selectedBubbleText: {
        color: 'white',
    },
    resetButton: {
        backgroundColor: '#f0f0f0',
        paddingVertical: 12,
        borderRadius: 8,
        alignItems: 'center',
        marginTop: 10,
    },
    resetButtonText: {
        color: '#333',
        fontWeight: '600',
    },
    clearDataButton: {
        backgroundColor: '#ff3b30',
        paddingVertical: 12,
        borderRadius: 8,
        alignItems: 'center',
        marginTop: 15,
    },
    clearDataButtonText: {
        color: 'white',
        fontWeight: '600',
    },
    resetModelButton: {
        backgroundColor: '#007AFF',
        paddingVertical: 12,
        borderRadius: 8,
        alignItems: 'center',
        marginTop: 10,
    },
    resetModelButtonDisabled: {
        backgroundColor: '#a0c8f0',
        elevation: 1,
        shadowOpacity: 0.1,
    },
    resetModelButtonText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: 'bold',
    },
    modelStatusContainer: {
        marginVertical: 15,
        paddingVertical: 10,
        borderTopWidth: 1,
        borderTopColor: '#eee',
        borderBottomWidth: 1,
        borderBottomColor: '#eee',
    },
    modelStatusLabel: {
        fontSize: 16,
        fontWeight: 'bold',
        marginBottom: 8,
    },
    modelStatusBadge: {
        paddingVertical: 6,
        paddingHorizontal: 12,
        borderRadius: 16,
        alignSelf: 'flex-start',
    },
    modelStatusOriginalBadge: {
        backgroundColor: '#e0e0e0',
    },
    modelStatusTrainedBadge: {
        backgroundColor: '#a5d6a7',
    },
    modelStatusText: {
        fontSize: 14,
        fontWeight: '500',
        color: '#333',
    },
    modelStatsContainer: {
        marginVertical: 15,
        paddingVertical: 15,
        paddingHorizontal: 10,
        backgroundColor: '#f5f5f5',
        borderRadius: 8,
        width: '100%',
        borderWidth: 1,
        borderColor: '#e0e0e0',
    },
    modelStatsTitle: {
        fontSize: 18,
        fontWeight: 'bold',
        marginBottom: 10,
        color: '#333',
        textAlign: 'center',
    },
    modelStatsText: {
        fontSize: 16,
        color: '#444',
        marginBottom: 15,
        textAlign: 'center',
    },
    modelStatsSubtitle: {
        fontSize: 14,
        fontWeight: 'bold',
        marginTop: 5,
        marginBottom: 8,
        color: '#555',
    },
    modelStatsItem: {
        fontSize: 14,
        color: '#666',
        marginBottom: 3,
        paddingLeft: 10,
    },
    loadingCardContent: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    loadingIconContainer: {
        marginBottom: 20,
    },
    loadingCardTitle: {
        fontSize: 22,
        fontWeight: 'bold',
        marginBottom: 12,
        color: '#333',
        textAlign: 'center',
    },
    loadingCardSubtitle: {
        fontSize: 16,
        color: '#666',
        textAlign: 'center',
        maxWidth: '80%',
        lineHeight: 22,
    },
    loadingCircle: {
        width: 100,
        height: 100,
        borderRadius: 50,
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.15,
        shadowRadius: 5,
        elevation: 3,
        borderWidth: 1,
        borderColor: 'rgba(0, 0, 0, 0.05)',
    },
});

export default SwipeScreen;
