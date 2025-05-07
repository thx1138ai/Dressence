/**
 * Preference storage and management service for user outfit swipes.
 *
 * - PreferenceService: Handles persistent storage of swipe preferences with AsyncStorage.
 *   - Maintains a rolling window of the most recent 200 preferences.
 *   - Tracks swipe statistics (total ever, since last training, last trained timestamp).
 *   - Determines when to trigger model retraining (after 50 swipes initially, then every 25 swipes).
 *
 * - Core Methods:
 *   - getPreferences(): Retrieve all stored preferences.
 *   - addPreference(pref): Save a new swipe, trim to the latest 200, update stats, and report if retraining is due.
 *   - markTrained(): Reset the swipesSinceLastTrain counter and update lastTrainedAt timestamp.
 *   - getTrainingStats()/saveTrainingStats(): Load and persist training metadata.
 *   - clearPreferences(): Wipe all stored preferences and stats.
 *   - resetPreferencesAndStats(): Fully reset storage to initial empty state.
 *   - getPreferenceCount(): Return current count of stored preferences.
 *   - updateConfidenceScores(predictor): Batch-process saved preferences missing confidence, fetch new scores, and persist.
*/

import AsyncStorage from '@react-native-async-storage/async-storage';
import { OutfitTensor } from './outfitGenerator';

const PREFERENCES_STORAGE_KEY = '@userOutfitPreferences_v2';
const MAX_STORED_PREFERENCES = 200; // Rolling window
const PREFERENCES_STATS_KEY = '@preferenceStats_v1';

export interface StoredPreference {
    tensor: OutfitTensor;
    temperature: number;
    liked: boolean;
    timestamp: string;
    confidence?: number;
}

export interface PreferenceStats {
    totalSwipesEver: number;
    swipesSinceLastTrain: number;
    lastTrainedAt: string | null;
}

export class PreferenceService {
    private updating = false;

    /**
     * Retrieves all stored preferences.
     */
    async getPreferences(): Promise<StoredPreference[]> {
        try {
            const prefsString = await AsyncStorage.getItem(PREFERENCES_STORAGE_KEY);
            return prefsString ? JSON.parse(prefsString) : [];
        } catch (error) {
            console.error('Error retrieving preferences from AsyncStorage:', error);
            return [];
        }
    }

    /**
     * Adds a new preference, implementing the rolling window of 200 preferences.
     */
    async addPreference(preference: StoredPreference): Promise<{shouldTrain: boolean, totalCount: number}> {
        try {
            const existingPrefs = await this.getPreferences();
            // Add new preference
            existingPrefs.push(preference);

            // Keep only the most recent 200 preferences
            const limitedPrefs = existingPrefs
                .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
                .slice(0, MAX_STORED_PREFERENCES);

            await AsyncStorage.setItem(PREFERENCES_STORAGE_KEY, JSON.stringify(limitedPrefs));
            
            // Update training stats
            const stats = await this.getTrainingStats();
            stats.totalSwipesEver += 1;
            stats.swipesSinceLastTrain += 1;
            await this.saveTrainingStats(stats);
            
            console.log(`Preference saved. Total stored: ${limitedPrefs.length}, Total ever: ${stats.totalSwipesEver}, Since last train: ${stats.swipesSinceLastTrain}`);
            
            // Determine if training should happen now based on the rules:
            // 1. At least 50 total swipes and never trained before
            // 2. Every 25 swipes after initial training
            const shouldTrain = 
                (stats.totalSwipesEver >= 50 && stats.lastTrainedAt === null) || 
                (stats.lastTrainedAt !== null && stats.swipesSinceLastTrain >= 25);
                
            return { 
                shouldTrain,
                totalCount: stats.totalSwipesEver
            };
        } catch (error) {
            console.error('Error saving preference to AsyncStorage:', error);
            return { shouldTrain: false, totalCount: 0 };
        }
    }

    /**
     * Mark that training has occurred
     */
    async markTrained(): Promise<void> {
        const stats = await this.getTrainingStats();
        stats.lastTrainedAt = new Date().toISOString();
        stats.swipesSinceLastTrain = 0;
        await this.saveTrainingStats(stats);
        console.log('Training stats updated');
    }

    /**
     * Get training statistics
     */
    async getTrainingStats(): Promise<PreferenceStats> {
        try {
            const statsString = await AsyncStorage.getItem(PREFERENCES_STATS_KEY);
            return statsString ? JSON.parse(statsString) : {
                totalSwipesEver: 0,
                swipesSinceLastTrain: 0,
                lastTrainedAt: null
            };
        } catch (error) {
            console.error('Error retrieving training stats:', error);
            return {
                totalSwipesEver: 0,
                swipesSinceLastTrain: 0,
                lastTrainedAt: null
            };
        }
    }

    /**
     * Save training statistics
     */
    private async saveTrainingStats(stats: PreferenceStats): Promise<void> {
        try {
            await AsyncStorage.setItem(PREFERENCES_STATS_KEY, JSON.stringify(stats));
        } catch (error) {
            console.error('Error saving training stats:', error);
        }
    }

    /**
     * Clears all stored preferences and resets training stats.
     */
    async clearPreferences(): Promise<void> {
        try {
            await AsyncStorage.removeItem(PREFERENCES_STORAGE_KEY);
            await AsyncStorage.removeItem(PREFERENCES_STATS_KEY);
            console.log('All preferences and training stats cleared.');
        } catch (error) {
            console.error('Error clearing preferences from AsyncStorage:', error);
        }
    }

    /**
     * Completely resets all preference data and training stats.
     * Unlike clearPreferences, this method explicitly sets empty values rather than just removing.
     */
    async resetPreferencesAndStats(): Promise<void> {
        try {
            // First clear existing data
            await AsyncStorage.removeItem(PREFERENCES_STORAGE_KEY);
            await AsyncStorage.removeItem(PREFERENCES_STATS_KEY);
            
            // Then explicitly set empty values
            await AsyncStorage.setItem(PREFERENCES_STORAGE_KEY, JSON.stringify([]));
            
            // Reset stats to initial values
            const initialStats: PreferenceStats = {
                totalSwipesEver: 0,
                swipesSinceLastTrain: 0,
                lastTrainedAt: null
            };
            await AsyncStorage.setItem(PREFERENCES_STATS_KEY, JSON.stringify(initialStats));
            
            console.log('All preferences and training stats fully reset to initial values.');
        } catch (error) {
            console.error('Error resetting preferences and stats in AsyncStorage:', error);
        }
    }

    /**
     * Gets the current count of stored preferences.
     */
    async getPreferenceCount(): Promise<number> {
        const prefs = await this.getPreferences();
        return prefs.length;
    }
    
    /**
     * Updates confidence scores for preferences that don't have them
     * Processes predictions in parallel batches for better performance
     * @param predictor Function that returns a prediction for a tensor and temperature
     * @returns Object containing count of updated items and total items
     */
    async updateConfidenceScores(
        predictor: (tensor: OutfitTensor, temperature: number) => Promise<number>
    ): Promise<{updated: number, total: number}> {
        // Prevent concurrent calls
        if (this.updating) {
            return { updated: 0, total: 0 };
        }
        
        this.updating = true;
        try {
            // Get all preferences
            const prefs = await this.getPreferences();
            let updatedCount = 0;
            
            // Find preferences that need confidence scores
            const missingIndices = prefs
                .map((p, i) => p.confidence === undefined ? i : -1)
                .filter(i => i >= 0);
            
            if (missingIndices.length === 0) {
                return { updated: 0, total: prefs.length };
            }
            
            // Process in batches of 25 to avoid memory issues
            const BATCH_SIZE = 25;
            for (let start = 0; start < missingIndices.length; start += BATCH_SIZE) {
                const batchIndices = missingIndices.slice(start, start + BATCH_SIZE);
                
                // Create promises for this batch
                const promises = batchIndices.map(i => 
                    predictor(prefs[i].tensor, prefs[i].temperature)
                        .then(confidence => {
                            prefs[i].confidence = confidence;
                            return i;
                        })
                        .catch(error => {
                            console.warn(`Failed to calculate confidence for preference ${i}:`, error);
                            return -1;
                        })
                );
                
                // Wait for all predictions in this batch
                const results = await Promise.all(promises);
                const successfulUpdates = results.filter(i => i >= 0).length;
                updatedCount += successfulUpdates;
                
                // Save after each batch for resilience
                if (successfulUpdates > 0) {
                    await AsyncStorage.setItem(PREFERENCES_STORAGE_KEY, JSON.stringify(prefs));
                    console.log(`Saved batch of ${successfulUpdates} confidence updates`);
                }
            }
            
            return { 
                updated: updatedCount,
                total: prefs.length
            };
        } catch (error) {
            console.error('Error updating confidence scores:', error);
            return { updated: 0, total: 0 };
        } finally {
            this.updating = false;
        }
    }
}
