# Architecture

## Overview

The Emotion Detection Service is structured as a modular C++ system composed of clearly separated layers:

1. **Service Layer (gRPC)**
2. **NLP Pipeline**
3. **Model Execution Layer**

The architecture follows a separation-of-concerns approach, where communication, preprocessing, and model inference are isolated into independent components.

---

## High-Level Flow

Text request → Preprocessing → Feature Vector (TF-IDF) → Model Inference → Label Mapping → Response

---

## Component Responsibilities

### 1. EmotionService (gRPC Layer)

- Receives text classification requests
- Validates input
- Delegates processing to the NLP pipeline
- Returns structured emotion scores

This layer contains no NLP logic — it is responsible strictly for communication and request handling.

---

### 2. NLPPipeline

The central orchestration component of the system.

Responsibilities:
- Executes preprocessing
- Passes feature vectors to the model
- Maps raw model outputs to emotion labels
- Returns final classification results

The pipeline encapsulates the full prediction workflow, keeping higher layers independent from implementation details.

---

### 3. ClassicalPreprocessor

Implements the full text preprocessing pipeline:

- Lowercasing
- Punctuation handling
- Tokenization
- Stopword removal
- Stemming
- N-gram generation
- TF-IDF vectorization

The output is a fixed-size numerical vector compatible with the trained model.

The preprocessing logic is implemented entirely within the project and does not rely on external NLP frameworks.

---

### 4. ONNXModel

Responsible for:

- Loading the exported ONNX model
- Executing inference using ONNX Runtime
- Returning raw prediction scores

This layer is isolated from preprocessing and service logic, making model replacement straightforward.

---

### 5. LabelMapping

- Translates raw model outputs into human-readable emotion labels
- Applies score normalization when required
- Formats results for the service response
