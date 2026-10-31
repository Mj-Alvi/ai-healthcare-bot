# Android APK for AI Healthcare Assistant

This directory contains the complete native Android WebView application project that wraps your deployed AI Healthcare Assistant into an Android APK.

## 🚀 3 Ways to Get Your Android APK

### Option 1: Instant 1-Click Online Generator (Recommended - No Android Studio Needed)
Once your web app is deployed to Streamlit Community Cloud (e.g. `https://my-healthcare-bot.streamlit.app`):
1. Go to **[WebIntoApp.com](https://www.webintoapp.com)** or **[PWABuilder.com](https://www.pwabuilder.com)**.
2. Enter your deployed URL (e.g., `https://your-bot.streamlit.app`).
3. Set App Name: `AI Healthcare Assistant`.
4. Click **Make App / Download APK**.
5. Transfer the `.apk` to your phone and tap to install!

---

### Option 2: Progressive Web App (PWA) - Direct Install (No APK Needed!)
1. Open your app URL in **Google Chrome** on your Android phone.
2. Tap the 3 dots menu `⋮` in the top right corner.
3. Tap **"Install App"** (or **"Add to Home screen"**).
4. The app icon will appear on your phone home screen and open full-screen like a native app.

---

### Option 3: Build from this Android Studio Project
If you have Android Studio installed on your computer:
1. Open **Android Studio**.
2. Click **File > Open** and select the `android_app` folder.
3. In `MainActivity.java`, update the `APP_URL` variable with your live URL:
   ```java
   public static final String APP_URL = "https://your-bot-name.streamlit.app";
   ```
4. Click **Build > Build Bundle(s) / APK(s) > Build APK(s)**.
5. Android Studio will compile `app-debug.apk` located at:
   `android_app/app/build/outputs/apk/debug/app-debug.apk`
