/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 */

import React from 'react';
import type {PropsWithChildren} from 'react';
import { LogBox } from 'react-native';
import SwipeScreen from './screens/SwipeScreen';
import WardrobeScreen from './screens/WardrobeScreen';
import OutfitGeneratorScreen from './screens/OutfitGeneratorScreen';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  useColorScheme,
  View,
} from 'react-native';

// Suppress TensorFlow.js kernel registration warnings
LogBox.ignoreLogs([
    /The kernel .*/,
    /cpu backend was already registered. Reusing existing backend factory/,
    />*/,
]);

import {
  Colors,
  DebugInstructions,
  Header,
  LearnMoreLinks,
  ReloadInstructions,
} from 'react-native/Libraries/NewAppScreen';

type SectionProps = PropsWithChildren<{
  title: string;
}>;

type RootStackParamList = {
  Swipe: undefined;
  Wardrobe: undefined;
  OutfitGenerator: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

function Section({children, title}: SectionProps): React.JSX.Element {
  const isDarkMode = false;
  // const isDarkMode = useColorScheme() === 'dark';
  return (
    <View style={styles.sectionContainer}>
      <Text
        style={[
          styles.sectionTitle,
          {
            color: isDarkMode ? Colors.white : Colors.black,
          },
        ]}>
        {title}
      </Text>
      <Text
        style={[
          styles.sectionDescription,
          {
            color: isDarkMode ? Colors.light : Colors.dark,
          },
        ]}>
        {children}
      </Text>
    </View>
  );
}

function App(): React.JSX.Element {
    const isDarkMode = useColorScheme() === 'dark';
    const barStyle = isDarkMode ? 'light-content' : 'dark-content';
  
    const backgroundStyle = {
      backgroundColor: isDarkMode ? Colors.darker : Colors.lighter,
      flex: 1,
    };
  
    return (
        <NavigationContainer>
          {/* Remove the top-level SafeAreaView here */}
          <StatusBar barStyle={barStyle} />
          <Stack.Navigator
            screenOptions={{
              // Make your header background white, if you want
              headerStyle: { backgroundColor: '#fff' },
              // Ensure header text/icons contrast as needed
              headerTintColor: '#000',
              // Make the screen content background white
              contentStyle: { backgroundColor: '#fff' },
            }}
          >
            <Stack.Screen
              name="Swipe"
              component={SwipeScreen}
              options={{ headerShown: false }}
            />
            <Stack.Screen
              name="Wardrobe"
              component={WardrobeScreen}
              // Possibly show the header; background is already set to white above
              options={{ headerShown: true }}
            />
            <Stack.Screen
              name="OutfitGenerator"
              component={OutfitGeneratorScreen}
              options={{ headerShown: false }}
            />
          </Stack.Navigator>
        </NavigationContainer>
    );
}

const styles = StyleSheet.create({
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '600',
  },
  sectionDescription: {
    marginTop: 8,
    fontSize: 18,
    fontWeight: '400',
  },
  highlight: {
    fontWeight: '700',
  },
});

export default App;
