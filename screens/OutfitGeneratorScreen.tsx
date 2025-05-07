/**
 * This is the screen that handles the outfit generation and viewing UI:
 *
 * - Loads the user's wardrobe items and any previously saved outfits from AsyncStorage on mount.
 * - Manages state for generation parameters (temperature), loading indicators, and view mode (generate vs saved).
 * - Calls the generateOutfit service to fetch AI-recommended outfits based on the selected temperature.
 * - Renders the generated outfit with images, and provides dismiss or save actions.
 * - Persists saved outfits along with metadata (timestamp, temperature) to AsyncStorage, and allows deletion.
 * - Offers a temperature slider with gradient track and custom thumb, and toggles parameter visibility.
 * - Supports navigation back to wardrobe or other screens via React Navigation.
*/

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
  Dimensions,
  Alert,
  FlatList,
  ImageSourcePropType,
  Platform
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import AsyncStorage from '@react-native-async-storage/async-storage';
import LinearGradient from 'react-native-linear-gradient';
import Slider from '@react-native-community/slider';
import { WardrobeItem, Outfit, generateOutfit, PrecomputedOutfit } from './services/outfitGenerator';

const IMAGE_ASSETS = {
  placeholder: require('./assets/icons/placeholder.jpg')
};

type RootStackParamList = {
  Swipe: undefined;
  Wardrobe: undefined;
  OutfitGenerator: undefined;
};

type OutfitGeneratorScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'OutfitGenerator'>;

type Props = {
  navigation: OutfitGeneratorScreenNavigationProp;
};

const TEMPERATURE_LABELS = ['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot'];
const SEASON_ICONS: { [key: string]: string } = {
  Spring: '🌸',
  Summer: '☀️',
  Fall: '🍂',
  Winter: '❄️',
};

const SCREEN_WIDTH = Dimensions.get('window').width;
const SCREEN_HEIGHT = Dimensions.get('window').height;

const OutfitGeneratorScreen: React.FC<Props> = ({ navigation }) => {
  // State variables for parameters
  const [temperature, setTemperature] = useState(2);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedOutfit, setGeneratedOutfit] = useState<Outfit | null>(null);
  const [wardrobeItems, setWardrobeItems] = useState<WardrobeItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [savedOutfits, setSavedOutfits] = useState<Outfit[]>([]);
  const [viewMode, setViewMode] = useState<'generator' | 'saved'>('generator');
  const [showParameters, setShowParameters] = useState(false);

  // Load wardrobe items and saved outfits when component mounts
  useEffect(() => {
    loadWardrobeItems();
    loadSavedOutfits();
  }, []);

  const loadWardrobeItems = async () => {
    try {
      setIsLoading(true);
      const savedItems = await AsyncStorage.getItem('wardrobeItems');
      if (savedItems) {
        const parsedItems = JSON.parse(savedItems) as WardrobeItem[];
        setWardrobeItems(parsedItems);
        console.log(`Loaded ${parsedItems.length} wardrobe items`);
      } else {
        console.log('No wardrobe items found');
      }
    } catch (error) {
      console.error('Error loading wardrobe items:', error);
      Alert.alert('Error', 'Failed to load your wardrobe items.');
    } finally {
      setIsLoading(false);
    }
  };

  const loadSavedOutfits = async () => {
    try {
      const savedOutfitsString = await AsyncStorage.getItem('savedOutfits');
      if (savedOutfitsString) {
        const parsedOutfits = JSON.parse(savedOutfitsString) as Outfit[];
        setSavedOutfits(parsedOutfits);
        console.log(`Loaded ${parsedOutfits.length} saved outfits`);
      } else {
        console.log('No saved outfits found');
      }
    } catch (error) {
      console.error('Error loading saved outfits:', error);
    }
  };

  const saveOutfit = async () => {
    if (!generatedOutfit) return;
    
    try {
      // Add timestamp to the outfit for sorting/display purposes
      const outfitToSave = {
        ...generatedOutfit,
        savedAt: new Date().toISOString(),
        temperature: temperature
      };
      
      // Add to saved outfits
      const updatedOutfits = [...savedOutfits, outfitToSave];
      
      // Save to AsyncStorage
      await AsyncStorage.setItem('savedOutfits', JSON.stringify(updatedOutfits));
      
      // Update state
      setSavedOutfits(updatedOutfits);
      
      Alert.alert('Success', 'Outfit saved successfully!');
      setGeneratedOutfit(null); // Clear the current outfit after saving
    } catch (error) {
      console.error('Error saving outfit:', error);
      Alert.alert('Error', 'Failed to save the outfit. Please try again.');
    }
  };

  const dismissOutfit = () => {
    setGeneratedOutfit(null);
  };

  const handleGenerateOutfit = async () => {
    if (wardrobeItems.length === 0) {
      Alert.alert(
        'Empty Wardrobe', 
        'Please add some clothes to your wardrobe first.',
        [
          { text: 'Go to Wardrobe', onPress: () => navigation.navigate('Wardrobe') },
          { text: 'Cancel', style: 'cancel' }
        ]
      );
      return;
    }

    try {
      setIsGenerating(true);
      setGeneratedOutfit(null);

      console.log(`Generating outfit with temperature: ${temperature}`);
      
      const result = await generateOutfit(
        'generate',
        temperature,
        50,
        [] as PrecomputedOutfit[],
        wardrobeItems
      );

      if (result && result.bestOutfit) {
        setGeneratedOutfit(result.bestOutfit);
        console.log('Outfit generated successfully');
      } else {
        Alert.alert('Error', 'Could not generate a suitable outfit with your current wardrobe.');
      }
    } catch (error) {
      console.error('Error generating outfit:', error);
      Alert.alert('Error', 'Failed to generate outfit.');
    } finally {
      setIsGenerating(false);
    }
  };

  const getImageSource = (item: WardrobeItem | null): ImageSourcePropType | null => {
    if (!item) return null;
    
    const paths = [
      item.processedUri,
      item.uri,
      item.segmented_image
    ].filter(Boolean) as string[];
    
    if (paths.length === 0) {
      console.warn(`No image paths available for item ${item.id || item.item_id || item.class_name || 'unknown'}`);
      return null;
    }
    
    // Use the first valid path
    let path = paths[0]!;
    
    // Ensure it has the file:// prefix if it's a local path
    if (!path.startsWith('file://') && !path.startsWith('http')) {
      path = `file://${path}`;
    }
    
    return { uri: path };
  };

  const renderCategoryLabel = (category: string) => {
    return (
      <View style={styles.categoryLabel}>
        <Text style={styles.categoryLabelText} allowFontScaling={false}>{category}</Text>
      </View>
    );
  };

  const renderItemContext = (item: WardrobeItem | null) => {
    if (!item) return null;
    
    // Return empty view since we don't want to show any labels
    return null;
  };

  const renderOutfitActions = () => {
    return (
      <View style={styles.outfitActionsContainer}>
        <TouchableOpacity 
          style={styles.dismissButton}
          onPress={dismissOutfit}
        >
          <Text style={styles.dismissButtonText} allowFontScaling={false}>Dismiss</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={styles.saveButton}
          onPress={saveOutfit}
        >
          <Text style={styles.saveButtonText} allowFontScaling={false}>Save Outfit</Text>
        </TouchableOpacity>
      </View>
    );
  };

  const renderGeneratedOutfit = () => {
    if (!generatedOutfit) return null;

    return (
      <View style={styles.generatedOutfitContainer}>
        <Text style={styles.generatedOutfitTitle}>Your Generated Outfit</Text>

        <View style={styles.outfitGrid}>
          {/* Top row with Top and Outer */}
          <View style={styles.outfitRow}>
            {/* Top item */}
            {generatedOutfit.base && (
              <View style={styles.outfitItemBox}>
                <View style={styles.outfitItemLabelContainer}>
                  <Text style={styles.outfitItemLabel}>Top</Text>
                </View>
                <View style={styles.imageContainer}>
                  <Image 
                    source={getImageSource(generatedOutfit.base) || IMAGE_ASSETS.placeholder} 
                    style={styles.outfitItemImage} 
                    resizeMode="contain"
                  />
                </View>
              </View>
            )}
            
            {/* Outer item */}
            {generatedOutfit.outer && (
              <View style={styles.outfitItemBox}>
                <View style={styles.outfitItemLabelContainer}>
                  <Text style={styles.outfitItemLabel}>Outer</Text>
                </View>
                <View style={styles.imageContainer}>
                  <Image 
                    source={getImageSource(generatedOutfit.outer) || IMAGE_ASSETS.placeholder} 
                    style={styles.outfitItemImage} 
                    resizeMode="contain"
                  />
                </View>
              </View>
            )}
          </View>
          
          {/* Second row for Bottom only */}
          {generatedOutfit.bottom && (
            <View style={styles.outfitRowSingle}>
              <View style={[styles.outfitItemBox, styles.bottomItemBox]}>
                <View style={styles.outfitItemLabelContainer}>
                  <Text style={styles.outfitItemLabel}>Bottom</Text>
                </View>
                <View style={styles.imageContainer}>
                  <Image 
                    source={getImageSource(generatedOutfit.bottom) || IMAGE_ASSETS.placeholder} 
                    style={styles.outfitItemImage} 
                    resizeMode="contain"
                  />
                </View>
              </View>
            </View>
          )}
        </View>
        
        {renderOutfitActions()}
      </View>
    );
  };

  const deleteSavedOutfit = async (outfitIndex: number) => {
    try {
      const updatedOutfits = [...savedOutfits];
      updatedOutfits.splice(outfitIndex, 1);
      
      await AsyncStorage.setItem('savedOutfits', JSON.stringify(updatedOutfits));
      setSavedOutfits(updatedOutfits);
      
      Alert.alert('Success', 'Outfit deleted successfully');
    } catch (error) {
      console.error('Error deleting outfit:', error);
      Alert.alert('Error', 'Failed to delete outfit. Please try again.');
    }
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  };

  const renderSavedOutfitItem = ({ item, index }: { item: Outfit & { savedAt?: string, temperature?: number }, index: number }) => {
    return (
      <View style={styles.savedOutfitCard}>
        <View style={styles.savedOutfitHeader}>
          <Text style={styles.savedOutfitDate}>
            {item.savedAt ? formatDate(item.savedAt) : 'Saved Outfit'}
          </Text>
          <TouchableOpacity 
            style={styles.deleteOutfitButton}
            onPress={() => {
              Alert.alert(
                'Delete Outfit',
                'Are you sure you want to delete this outfit?',
                [
                  { text: 'Cancel', style: 'cancel' },
                  { text: 'Delete', style: 'destructive', onPress: () => deleteSavedOutfit(index) }
                ]
              );
            }}
          >
            <Text style={styles.deleteOutfitButtonText}>✕</Text>
          </TouchableOpacity>
        </View>
        
        <View style={styles.outfitGrid}>
          {/* Top row with Top and Outer */}
          <View style={styles.outfitRow}>
            {/* Top item */}
            {item.base && (
              <View style={styles.outfitItemBox}>
                <View style={styles.outfitItemLabelContainer}>
                  <Text style={styles.outfitItemLabel}>Top</Text>
                </View>
                <View style={styles.imageContainer}>
                  <Image 
                    source={getImageSource(item.base) || IMAGE_ASSETS.placeholder} 
                    style={styles.outfitItemImage} 
                    resizeMode="contain"
                  />
                </View>
              </View>
            )}
            
            {/* Outer item */}
            {item.outer && (
              <View style={styles.outfitItemBox}>
                <View style={styles.outfitItemLabelContainer}>
                  <Text style={styles.outfitItemLabel}>Outer</Text>
                </View>
                <View style={styles.imageContainer}>
                  <Image 
                    source={getImageSource(item.outer) || IMAGE_ASSETS.placeholder} 
                    style={styles.outfitItemImage} 
                    resizeMode="contain"
                  />
                </View>
              </View>
            )}
          </View>
          
          {/* Second row for Bottom only */}
          {item.bottom && (
            <View style={styles.outfitRowSingle}>
              <View style={[styles.outfitItemBox, styles.bottomItemBox]}>
                <View style={styles.outfitItemLabelContainer}>
                  <Text style={styles.outfitItemLabel}>Bottom</Text>
                </View>
                <View style={styles.imageContainer}>
                  <Image 
                    source={getImageSource(item.bottom) || IMAGE_ASSETS.placeholder} 
                    style={styles.outfitItemImage} 
                    resizeMode="contain"
                  />
                </View>
              </View>
            </View>
          )}
        </View>
        
        <View style={styles.savedOutfitDetails}>
          {item.temperature !== undefined && (
            <View style={styles.savedOutfitParameter}>
              <Text style={styles.savedOutfitParameterLabel}>Temperature:</Text>
              <Text style={styles.savedOutfitParameterValue}>
                {TEMPERATURE_LABELS[item.temperature]}
              </Text>
            </View>
          )}
        </View>
      </View>
    );
  };

  const renderSavedOutfits = () => {
    if (savedOutfits.length === 0) {
      return (
        <View style={styles.emptyStateContainer}>
          <Text style={styles.emptyStateText}>
            You haven't saved any outfits yet. Generate and save some outfits to see them here.
          </Text>
        </View>
      );
    }

    return (
      <FlatList
        data={savedOutfits}
        keyExtractor={(_, index) => `saved-outfit-${index}`}
        renderItem={renderSavedOutfitItem}
        contentContainerStyle={[styles.savedOutfitsList, styles.scrollContentContainer]}
        style={styles.scrollContainer}
      />
    );
  };

  const renderContent = () => {
    if (isLoading) {
      return (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#000" />
          <Text style={styles.loadingText}>Loading your wardrobe...</Text>
        </View>
      );
    }

    if (viewMode === 'generator') {
      return (
        <>
          <ScrollView style={styles.scrollContainer} contentContainerStyle={styles.scrollContentContainer}>
            {generatedOutfit && renderGeneratedOutfit()}
            
            {!generatedOutfit && !isGenerating && (
              <View style={styles.emptyStateContainer}>
                <Text style={styles.emptyStateText}>
                  Adjust the parameters below and tap "Generate Outfit" to create an outfit from your wardrobe.
                </Text>
              </View>
            )}
            
            {!generatedOutfit && isGenerating && (
              <View style={styles.loadingOutfitContainer}>
                <ActivityIndicator size="large" color="#007AFF" />
                <Text style={styles.loadingOutfitText}>Loading AI model and generating your outfit...</Text>
                <Text style={styles.loadingOutfitSubtext}>This may take a moment as we prepare the best outfit for you.</Text>
              </View>
            )}
          </ScrollView>

          <View style={styles.parametersContainer}>
            <TouchableOpacity 
              style={styles.parametersTitleRow}
              onPress={() => setShowParameters(!showParameters)}
            >
              <Text style={styles.parametersTitle}>Outfit Parameters</Text>
              <Text style={styles.expandCollapseIcon}>{showParameters ? '▼' : '▲'}</Text>
            </TouchableOpacity>
            
            {showParameters && (
              <View style={styles.parameterRow}>
                <Text style={styles.parameterRowLabel}>Temperature:</Text>
                <View style={styles.sliderContainer}>
                  <View style={styles.compactSliderRow}>
                    <Text style={styles.sliderMin}>Cold</Text>
                    <View style={styles.sliderWrapper}>
                      <LinearGradient
                        colors={['#6495ED', '#90EE90', '#FFD700', '#FFA500', '#FF4500']}
                        start={{x: 0, y: 0.5}}
                        end={{x: 1, y: 0.5}}
                        style={styles.sliderTrack}
                      />
                      <View style={styles.markersContainer}>
                        {[0, 1, 2, 3, 4].map((mark) => (
                          <View 
                            key={`mark-${mark}`} 
                            style={[
                              styles.sliderMarker, 
                              temperature === mark && styles.sliderMarkerActive
                            ]} 
                          />
                        ))}
                      </View>
                      <Slider
                        style={styles.slider}
                        minimumValue={0}
                        maximumValue={4}
                        step={1}
                        value={temperature}
                        onValueChange={setTemperature}
                        minimumTrackTintColor="transparent"
                        maximumTrackTintColor="transparent"
                        thumbTintColor="transparent"
                      />
                      <View style={[
                        styles.customThumb, 
                        { left: `${(temperature / 4) * 100}%` }
                      ]}>
                        <View style={styles.thumbInner} />
                      </View>
                    </View>
                    <Text style={styles.sliderMax}>Hot</Text>
                  </View>
                  <Text style={styles.parameterValue}>{TEMPERATURE_LABELS[temperature]}</Text>
                </View>
              </View>
            )}
            
            <TouchableOpacity 
              style={[styles.generateButton, isGenerating && styles.generateButtonDisabled]}
              onPress={handleGenerateOutfit}
              disabled={isGenerating}
            >
              {isGenerating ? (
                <View style={styles.loadingButtonContent}>
                  <ActivityIndicator size="small" color="#FFFFFF" />
                  <Text style={styles.generateButtonText}>Loading Model & Generating...</Text>
                </View>
              ) : (
                <Text style={styles.generateButtonText}>Generate Outfit</Text>
              )}
            </TouchableOpacity>
          </View>
        </>
      );
    } else {
      return renderSavedOutfits();
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <View style={styles.header}>
          <View style={styles.headerLeftSection}>
            <TouchableOpacity 
              style={styles.backButton}
              onPress={() => navigation.goBack()}
            >
              <Text style={styles.backButtonText}>←</Text>
            </TouchableOpacity>
            <Text style={styles.headerTitle} allowFontScaling={false}>Outfit Generator</Text>
          </View>
        </View>

        <View style={styles.viewModeSwitcher}>
          <TouchableOpacity 
            style={[
              styles.viewModeButton, 
              viewMode === 'generator' && styles.viewModeButtonActive
            ]}
            onPress={() => setViewMode('generator')}
          >
            <Text style={[
              styles.viewModeButtonText,
              viewMode === 'generator' && styles.viewModeButtonTextActive
            ]}>
              Generate
            </Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={[
              styles.viewModeButton, 
              viewMode === 'saved' && styles.viewModeButtonActive
            ]}
            onPress={() => setViewMode('saved')}
          >
            <Text style={[
              styles.viewModeButtonText,
              viewMode === 'saved' && styles.viewModeButtonTextActive
            ]}>
              Saved Outfits {savedOutfits.length > 0 ? `(${savedOutfits.length})` : ''}
            </Text>
          </TouchableOpacity>
        </View>

        <View style={styles.mainContentContainer}>
          {renderContent()}
        </View>
      </View>
    </SafeAreaView>
  );
};

const { width } = Dimensions.get('window');
const gridItemWidth = (width - 60) / 2;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    paddingBottom: Platform.OS === 'ios' ? 20 : 0,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 0 : 16,
    paddingBottom: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerTitle: {
    fontSize: Math.min(20, SCREEN_WIDTH * 0.05),
    fontWeight: 'bold',
    color: '#333',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#555',
  },
  parametersContainer: {
    backgroundColor: '#f8f8f8',
    borderRadius: 12,
    padding: 12,
    marginTop: 5,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#eaeaea',
  },
  parametersTitleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  parametersTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  expandCollapseIcon: {
    fontSize: 14,
    color: '#555',
  },
  parameterRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  parameterRowLabel: {
    width: 90,
    fontSize: 14,
    fontWeight: '500',
    color: '#444',
  },
  sliderContainer: {
    flex: 1,
  },
  compactSliderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 28,
  },
  sliderMin: {
    width: 40,
    fontSize: 10,
    color: '#555',
    marginRight: 2,
  },
  sliderMax: {
    width: 40,
    fontSize: 10,
    color: '#555',
    marginLeft: 2,
    textAlign: 'right',
  },
  sliderWrapper: {
    flex: 1,
    position: 'relative',
    height: 40,
  },
  sliderTrack: {
    position: 'absolute',
    top: 13,
    left: 0,
    right: 0,
    height: 8,
    borderRadius: 4,
    zIndex: 1,
  },
  slider: {
    position: 'absolute',
    width: '100%',
    height: 40,
    top: 0,
    zIndex: 10,
  },
  markersContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: '1.5%',
    zIndex: 2,
  },
  sliderMarker: {
    width: 4,
    height: 16,
    borderRadius: 2,
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    marginTop: 9,
  },
  sliderMarkerActive: {
    backgroundColor: '#fff',
    height: 20,
    marginTop: 7,
  },
  customThumb: {
    position: 'absolute',
    top: 5,
    marginLeft: -15,
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 3,
    elevation: 4,
    zIndex: 4,
  },
  thumbInner: {
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: '#333',
  },
  parameterValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    textAlign: 'center',
    marginTop: 8,
  },
  outfitActionsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: '5%',
    marginTop: 16,
    marginBottom: Platform.OS === 'ios' ? 30 : 20,
  },
  dismissButton: {
    backgroundColor: '#f0f0f0',
    paddingVertical: Math.min(15, SCREEN_WIDTH * 0.035),
    paddingHorizontal: Math.min(20, SCREEN_WIDTH * 0.05),
    borderRadius: 25,
    width: SCREEN_WIDTH * 0.4,
    maxWidth: 160,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  dismissButtonText: {
    fontSize: Math.min(16, SCREEN_WIDTH * 0.038),
    fontWeight: '600',
    color: '#555',
  },
  saveButton: {
    backgroundColor: '#4CAF50',
    paddingVertical: Math.min(15, SCREEN_WIDTH * 0.035),
    paddingHorizontal: Math.min(20, SCREEN_WIDTH * 0.05),
    borderRadius: 25,
    width: SCREEN_WIDTH * 0.4,
    maxWidth: 160,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 3,
  },
  saveButtonText: {
    fontSize: Math.min(16, SCREEN_WIDTH * 0.038),
    fontWeight: '600',
    color: 'white',
  },
  sliderThumb: {
    width: 28,
    height: 28,
    backgroundColor: '#3333FF',
    borderRadius: 14,
    borderWidth: 2,
    borderColor: '#fff',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 2,
    elevation: 3,
  },
  generateButton: {
    backgroundColor: '#333',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  generateButtonDisabled: {
    opacity: 0.8,
    backgroundColor: '#0056b3',
  },
  generateButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  emptyStateContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    marginBottom: 20,
  },
  emptyStateText: {
    fontSize: 16,
    color: '#777',
    textAlign: 'center',
    lineHeight: 24,
  },
  generatedOutfitContainer: {
    backgroundColor: '#f8f8f8',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#eaeaea',
  },
  generatedOutfitTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 16,
    color: '#333',
    textAlign: 'center',
  },
  outfitItemContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    width: SCREEN_WIDTH * 0.85,
    maxWidth: 360,
    height: Math.min(SCREEN_WIDTH * 0.75, 320),
    borderRadius: 16,
    backgroundColor: '#f5f5f5',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    position: 'relative',
    overflow: 'hidden',
  },
  imageContainer: {
    height: 180,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 8,
  },
  outfitItemImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  },
  categoryLabel: {
    position: 'absolute',
    top: 10,
    left: 10,
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 12,
    zIndex: 10,
  },
  categoryLabelText: {
    color: 'white',
    fontSize: Math.min(12, SCREEN_WIDTH * 0.03),
    fontWeight: '600',
  },
  itemContextContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 8,
    gap: 6,
  },
  contextTag: {
    backgroundColor: '#fff',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  contextTagText: {
    fontSize: 10,
    color: '#555',
  },
  outfitDetailsButton: {
    marginTop: 15,
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  outfitDetailsButtonText: {
    color: '#333',
    fontSize: 14,
    fontWeight: '500',
  },
  outfitDetails: {
    marginTop: 15,
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 15,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  outfitDetailsTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 10,
    color: '#333',
  },
  outfitParametersContainer: {
    marginTop: 10,
  },
  parameterContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  parameterLabel: {
    fontSize: 14,
    color: '#555',
  },
  headerLeftSection: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  backButton: {
    backgroundColor: '#f0f0f0',
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 10,
  },
  backButtonText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  viewModeSwitcher: {
    flexDirection: 'row',
    marginBottom: 20,
    borderRadius: 10,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  viewModeButton: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
    backgroundColor: '#f8f8f8',
  },
  viewModeButtonActive: {
    backgroundColor: '#333',
  },
  viewModeButtonText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#555',
  },
  viewModeButtonTextActive: {
    color: '#fff',
  },
  savedOutfitsList: {
    paddingBottom: 20,
  },
  savedOutfitCard: {
    backgroundColor: '#f8f8f8',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#eaeaea',
  },
  savedOutfitHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  savedOutfitDate: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  deleteOutfitButton: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#f0f0f0',
    alignItems: 'center',
    justifyContent: 'center',
  },
  deleteOutfitButtonText: {
    fontSize: 14,
    color: '#ff3b30',
    fontWeight: 'bold',
  },
  savedOutfitDetails: {
    marginTop: 15,
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  savedOutfitParameter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  savedOutfitParameterLabel: {
    fontSize: 14,
    color: '#555',
  },
  savedOutfitParameterValue: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
  },
  outfitGrid: {
    width: '100%',
  },
  outfitRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  outfitRowSingle: {
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  outfitItemBox: {
    width: '48%',
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: 'white',
    borderWidth: 1,
    borderColor: '#eaeaea',
  },
  bottomItemBox: {
    width: '70%',
  },
  outfitItemLabelContainer: {
    backgroundColor: '#eaeaea',
    padding: 8,
  },
  outfitItemLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    textAlign: 'center',
  },
  mainContentContainer: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
  },
  scrollContainer: {
    flex: 1,
  },
  scrollContentContainer: {
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  loadingOutfitContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    minHeight: 300,
  },
  loadingOutfitText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginTop: 20,
    marginBottom: 10,
    textAlign: 'center',
  },
  loadingOutfitSubtext: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    paddingHorizontal: 20,
  },
  loadingButtonContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  actionButton: {
    backgroundColor: 'white',
    paddingVertical: Math.min(12, SCREEN_WIDTH * 0.03),
    paddingHorizontal: Math.min(16, SCREEN_WIDTH * 0.04),
    borderRadius: 25,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 4,
    marginHorizontal: 8,
    marginBottom: 16,
    width: SCREEN_WIDTH * 0.42,
    maxWidth: 180,
  },
  buttonText: {
    fontSize: Math.min(16, SCREEN_WIDTH * 0.04),
    fontWeight: '500',
    color: '#333',
    textAlign: 'center',
  },
});

export default OutfitGeneratorScreen;
