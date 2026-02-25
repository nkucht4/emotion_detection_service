# Preprocessing Module

The `ClassicalPreprocessor` is a core component of the NLP pipeline.  
It is responsible for converting raw text into a numerical representation suitable for model inference.  

---

## Pipeline Overview

Input text is processed in the following steps:

1. **Lowercasing**
   - Converts all characters to lowercase for consistency.
   
2. **Punctuation Handling**
   - Removes most punctuation, preserving only `!` and `?` as they may carry emotional information.
   
3. **Tokenization**
   - Splits text into individual words (tokens) using whitespace as delimiter.

4. **Stopword Removal**
   - Removes common, non-informative words based on a predefined stopword list.

5. **Stemming**
   - Reduces words to their root form by stripping common suffixes (e.g., `running → run`).

6. **N-Gram Generation**
   - Creates bigrams in addition to individual tokens to capture local context.

7. **TF-IDF Vectorization**
   - Converts tokens into a fixed-size numerical vector:
     - Term Frequency (TF) measures token occurrence in the document.
     - Inverse Document Frequency (IDF) weights tokens based on their overall rarity.
   - Normalizes scores for consistent input to the model.

---

## Key Features

- Fully implemented within the project; does not rely on external NLP frameworks.
- Output is compatible with the ONNX-based model.
- Modular design allows easy replacement or extension (e.g., adding new tokenization or normalization methods).
- Efficient memory usage using `Eigen::VectorXf` for TF-IDF vectors.

---

## Example Usage

```cpp
#include "nlp/preprocessing/ClassicalPreprocessor.hpp"

ClassicalPreprocessor preprocessor("vocab.txt", "idf.txt");

std::string input_text = "I am very happy today!";
Eigen::VectorXf features = preprocessor.preprocessToVector(input_text);
```

The features vector can then be passed directly to the ONNX model for inference.
