# Emotion Detection Service

C++ NLP service with modular architecture, designed for text-based emotion recognition. The project combines ONNX-based inference and gRPC.

## Features
- Text Emotion Recognition: Classifies text input into multiple emotion labels using a pre-trained ONNX model.
  - Note: The model has been trained on limited vocabulary so far, so the predictions may not be accurate.
- Modular C++ Architecture: Clean separation between model inference, service logic, and communication layers.
- gRPC Integration (WIP): Provides classification as a service.
- ONNX Runtime: Leverages ONNX for fast, platform-independent model inference.
- Extensive Testing: Comprehensive unit and integration tests ensure correctness and robustness.
- Configurable Parameters: Key settings such as model path and others can be adjusted without code changes.

## Technology stack
### Service & Inference
 - C++ 17
 - ONNX Runtime
 - gRPC
 - Eigen

### Build & Testing
 - CMake
 - Google Test

### Model training
- Python
- sklearn

## Architecture
```mermaid
classDiagram
    class EmotionService {
        +GetTextEmotion(context, request, response)
    }

    class NLPPipeline {
        +run(text)
    }

    class ClassicalPreprocessor {
        +preprocessToVector(text)
    }

    class LabelMapping {
        +map(scores)
    }

    class ONNXModel {
        +load(path)
        +predict(input_vector)
    }

    EmotionService --> NLPPipeline : uses
    NLPPipeline --> ClassicalPreprocessor : contains
    NLPPipeline --> LabelMapping : contains
    NLPPipeline --> ONNXModel : contains
  ```

## Project structure
