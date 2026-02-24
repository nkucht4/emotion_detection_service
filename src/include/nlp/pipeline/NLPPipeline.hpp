#pragma once

#include <string>
#include <unordered_map>

#include "nlp/preprocessing/ClassicalPreprocessor.hpp"
#include "nlp/model/ONNXModel.hpp"
#include "nlp/utils/LabelMapper.hpp"

class NLPPipeline{
public:
    NLPPipeline(const std::string &vocab_file, const std::string &idf_file, 
        const std::string& model_path, const std::vector<std::string>& labels);

    std::unordered_map<std::string, float> run(std::string text);

private:
    ClassicalPreprocessor preprocessor;
    ONNXModel model;
    LabelMapper mapper;
};