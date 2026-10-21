# Windows Setup and Deployment Guide
### Psychological State Analysis Using Multimodal AI
### University Academic Research Prototype

This step-by-step guide is tailored for running the prototype locally on a **Windows 10 or Windows 11 laptop with 16 GB RAM**.

---

## 1. System Prerequisites

1. **Operating System:** Windows 10 or Windows 11 (64-bit).
2. **System Memory (RAM):** 16 GB recommended (minimum 8 GB with pagefile enabled).
3. **Python Version:** Python 3.12 or 3.13 (64-bit).
   - Ensure **"Add Python to PATH"** was selected during installation.
4. **Hardware Permissions:**
   - Microphone access enabled in Windows Settings (for live audio recording).
   - Camera access enabled in Windows Settings (for webcam facial expression recognition).

---

## 2. Setting Up the Python Virtual Environment

Using a virtual environment isolates project packages and prevents conflicts with other Python installations.

### Step 2.1: Open PowerShell or Windows Terminal
Press `Win + X` and select **Windows Terminal** or **PowerShell**.

Navigate to the project root directory:
```powershell
cd "d:\ABM-CV-PROJECT & RESEARCH PAPER"
```

### Step 2.2: Enable PowerShell Script Execution (If Required)
If Windows restricts executing unsigned scripts, execute the following command for your current PowerShell session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step 2.3: Create the Virtual Environment
Create an isolated environment named `venv`:
```powershell
python -m venv venv
```

### Step 2.4: Activate the Virtual Environment
- **In PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(Your terminal prompt will now be prefixed with `(venv)`).*

- **In Command Prompt (cmd.exe):**
  ```cmd
  venv\Scripts\activate.bat
  ```

---

## 3. Installing Dependencies

With the virtual environment active, install the verified requirements:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Dependency Verification Check
Confirm that all core packages import cleanly:
```powershell
python -c "import torch, cv2, librosa, soundfile, sklearn, transformers, streamlit, plotly; print('All libraries verified successfully!')"
```

---

## 4. Running the Automated Test Suite

Before running the dashboard, execute the unit test suite to verify preprocessing, audio extraction, NLP pipelines, and fusion logic:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

**Expected Output:**
```
Ran 13 tests in ~5s
OK
```

---

## 5. Launching the Research Dashboard

Launch the interactive Streamlit application:

```powershell
streamlit run app.py
```

The terminal will display the local network URL:
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Open your web browser (Chrome, Edge, or Firefox) and navigate to `http://localhost:8501`.

---

## 6. Demonstration Workflow for University Evaluation

When presenting the prototype to your supervisor or review committee, follow this structured demonstration flow:

1. **Page 1: Overview & Pipeline**
   - Review the system architecture diagram showing the 3 independent streams converging on Russell's Circumplex Affective Space.
   - Highlight the **Mandatory Ethical & Academic Disclaimer** stating that this is an affective computing methodology prototype, not a medical diagnostic device.

2. **Page 2: Text Analysis**
   - Click the **"📄 Sample: Depressed Affect"** button.
   - Click **"🔍 Analyze Text Modality"**.
   - Show the RoBERTa emotion probability distribution (high sadness probability), the Pennebaker LIWC pronoun density card (>8% First-Person Singular Pronouns reflecting elevated self-focus), and the traditional baseline comparison.

3. **Page 3: Speech & Audio Analysis**
   - Click **"🎙️ Sample: Monotone / Flat Prosody"**.
   - Click **"🔍 Execute Dual-Path Speech Analysis"**.
   - Show **Path A** (Whisper ASR transcription & downstream NLP) and **Path B** (Librosa vocal prosody: Fundamental Frequency F0 contour, low pitch variability SD indicating flat affect, and RMS loudness dynamic range).

4. **Page 4: Facial Expression Analysis**
   - Click **"📷 Sample: Warm Smile / Joy"**.
   - Click **"🔍 Execute Facial Expression Recognition"**.
   - Point out OpenCV's bounding box and ViT classification showing a high joy/happy probability.
   - Explain that visible facial expressions reflect surface motor movements (Action Units) and can be masked.

5. **Page 5: Multimodal Comparison & Affective Dissonance**
   - Observe how the session state carried forward the **Sad text**, **Flat monotone audio**, and **Smiling face**.
   - Highlight the **Affective Dissonance Alert**: The system automatically detects the discrepancy between sad lexical sentiment and a smiling face (classic social masking / smiling depression phenomenon).
   - Inspect **Russell's 2D Circumplex Plot** showing the spatial coordinates of each modality and the fused centroid.
   - Review the **Text-Only vs. Multimodal Comparison Card** explaining how non-verbal channels modulated the lexical prediction.

6. **Page 6: Model Information & Evaluation**
   - Review the empirical benchmark table showing the test-set comparison between the traditional TF-IDF baseline (45% F1) and the neural transformer (89% F1) with confusion matrix heatmaps.
   - Review the public multimodal datasets guide (RAVDESS, CREMA-D, DAIC-WOZ) and the data leakage prevention protocols (speaker/participant-independent splits).

---

## 7. Troubleshooting Windows-Specific Issues

### 1. Port 8501 Already in Use
If another application or previous session is occupying port 8501:
```powershell
streamlit run app.py --server.port 8502
```

### 2. Windows Script Execution Policy Error
If PowerShell states that running scripts is disabled on this system:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Microphone or Webcam Permission Error
If Streamlit's `st.audio_input` or `st.camera_input` cannot access the hardware:
1. Open **Windows Settings** (`Win + I`).
2. Go to **Privacy & Security** &rarr; **Microphone** (and **Camera**).
3. Ensure **"Let desktop apps access your microphone/camera"** is switched **ON**.
4. In your browser (Chrome/Edge), grant microphone and camera permissions to `http://localhost:8501`.
5. Note: If no microphone or webcam is available, the prototype includes **preloaded sample audio and photo buttons** so all features can be tested offline without hardware.

### 4. Audio Decoding or SoundFile Error
The `soundfile` library bundles precompiled `libsndfile` 64-bit binaries for Windows automatically. If an unrecognized format error occurs, ensure the uploaded file is in `.wav` or `.mp3` format.

---

## 8. Summary of Project Files

- [app.py](file:///d:/ABM-CV-PROJECT%20&%20RESEARCH%20PAPER/app.py): Complete Streamlit web application.
- [config.py](file:///d:/ABM-CV-PROJECT%20&%20RESEARCH%20PAPER/config.py): Parameters, model paths, taxonomy mappings, and disclaimers.
- [modules/text_analysis.py](file:///d:/ABM-CV-PROJECT%20&%20RESEARCH%20PAPER/modules/text_analysis.py): RoBERTa emotion model & Pennebaker LIWC metrics.
- [modules/audio_features.py](file:///d:/ABM-CV-PROJECT%20&%20RESEARCH%20PAPER/modules/audio_features.py): Librosa acoustic feature extractor (F0, RMS, tempo, spectral).
- [modules/speech_analysis.py](file:///d:/ABM-CV-PROJECT%20&%20RESEARCH%20PAPER/modules/speech_analysis.py): Whisper ASR transcription & prosody coordinator.
- [modules/face_analysis.py](file:///d:/ABM-CV-PROJECT%20&%20RESEARCH%20PAPER/modules/face_analysis.py): OpenCV face detection & ViT expression classifier.
- [modules/multimodal_fusion.py](file:///d:/ABM-CV-PROJECT%20&%20RESEARCH%20PAPER/modules/multimodal_fusion.py): Late fusion, Russell Circumplex, & dissonance engine.
- [modules/dataset_evaluation.py](file:///d:/ABM-CV-PROJECT%20&%20RESEARCH%20PAPER/modules/dataset_evaluation.py): Baseline training, metrics, & dataset guides.
- [utils/preprocessing.py](file:///d:/ABM-CV-PROJECT%20&%20RESEARCH%20PAPER/utils/preprocessing.py): Conditioning routines for text, audio, and face crops.
- [utils/visualizer.py](file:///d:/ABM-CV-PROJECT%20&%20RESEARCH%20PAPER/utils/visualizer.py): Interactive Plotly chart builders.
- [tests/](file:///d:/ABM-CV-PROJECT%20&%20RESEARCH%20PAPER/tests): Comprehensive automated unit test suite.
