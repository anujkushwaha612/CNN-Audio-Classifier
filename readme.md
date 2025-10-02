# 🔊 Deep Audio Classification CNN (PyTorch & Next.js)

This project is a full-stack, end-to-end solution for **Deep Audio Classification**. It involves building and training a complex Convolutional Neural Network (CNN) from scratch in PyTorch to classify diverse environmental sounds from raw audio waveforms, followed by deploying the model via a serverless API and creating an interactive dashboard for real-time visualization and inference.

---

## ✨ Key Features & Technical Stack

### Core Model & Training
* **Deep Audio CNN** engineered from scratch in **PyTorch**.
* **ResNet-style architecture** with residual blocks for high-precision classification.
* **Mel Spectrograms** utilized for audio-to-image conversion to feed the CNN.
* Advanced **Data Augmentation** techniques: Mixup & Time/Frequency Masking.
* Optimized training pipeline using **AdamW** and **OneCycleLR** scheduler.
* **TensorBoard** integration for detailed training analysis and monitoring.

### Deployment & Backend API
* Serverless **GPU inference** deployed via **Modal**.
* High-performance **FastAPI** endpoint for inference requests.
* **Pydantic** data validation for robust and secure API requests.

### Frontend Dashboard & Visualization
* Interactive dashboard built with **Next.js**, **React**, and **Tailwind CSS** (T3 Stack).
* **Real-time** audio classification with confidence score display.
* **Dynamic Visualization** of internal CNN feature maps (to see what the model "sees").
* Waveform and Spectrogram visualization for input analysis.

---

## 🛠️ Project Setup Instructions

To replicate the project, follow these steps.

### 1. Initial Clone & Environment

| Step | Command |
| :--- | :--- |
| **Clone Repository** | `git clone https://github.com/Andreaswt/audio-cnn.git` |
| **Navigate to Root** | `cd audio-cnn` |
| **Setup Python** | *(Ensure a virtual environment with Python 3.12 is active.)* |

### 2. Backend (PyTorch Model & API)

| Step | Command | Description |
| :--- | :--- | :--- |
| **Install Dependencies** | `pip install -r requirements.txt` | Installs PyTorch, FastAPI, Modal, etc. |
| **Configure Modal** | `modal setup` | Sets up serverless GPU deployment environment. |
| **Deploy Inference API** | `modal deploy main.py` | Deploys the trained model and FastAPI endpoint to Modal. |

### 3. Frontend (Next.js Dashboard)

| Step | Command | Description |
| :--- | :--- | :--- |
| **Navigate to Frontend** | `cd audio-cnn-visualisation` | |
| **Install Node Dependencies** | `npm i` | Installs Next.js, React, and other dashboard packages. |
| **Run Development Server** | `npm run dev` | Starts the interactive dashboard locally. |