# Dressence Fashion Recommendation App

This is the repository for the mobile app developed as part of my final-year dissertation at UCL, titled:

**"Dressence: Personalised Fashion Styling. An On-Device AI-Powered Approach to Style Recommendations"**

The app performs on-device inference to deliver fashion recommendations based on user preferences and contextual factors. It has a recommendation engine built with TensorFlow.js and runs entirely offline on mobile devices.

## Run App on Emulator

### iOS

```bash
npm install
```

Then:

```bash
cd ios
pod install
open Strut.xcworkspace
```

From Xcode, run the app on the simulator or your connected iPhone.

Metro server start:
```bash
npx expo start
```

---

## Run on Real iPhone

```bash
cd ios
pod install
open Strut.xcworkspace
```

Make sure your iPhone is:

- Connected via USB  
- In developer mode  
- Selected as the target device in Xcode  

Then run the app from Xcode.
