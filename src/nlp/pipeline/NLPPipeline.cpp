#include "nlp/pipeline/NLPPipeline.hpp"
#include "nlp/utils/Normalization.hpp"
#include <iostream>

#include <Eigen/Dense>

NLPPipeline::NLPPipeline(const std::string &vocab_file, const std::string &idf_file, 
        const std::string& model_path, const std::vector<std::string>& labels)
        : preprocessor{vocab_file, idf_file}, mapper{labels}{
            model = ONNXModel();
            model.load(model_path);
        }

std::unordered_map<std::string, float> NLPPipeline::run(std::string text){
    if (text.empty())
        throw std::invalid_argument("Input text is empty");
        
    auto input_vector = preprocessor.preprocessToVector(text);

    auto prediction = model.predict(input_vector);
    auto normalized_prediction = softmax(prediction);
    return mapper.map(prediction);
}