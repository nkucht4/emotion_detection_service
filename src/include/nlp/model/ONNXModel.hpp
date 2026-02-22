#pragma once

#include "IModel.hpp"

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>

class ONNXModel : public IModel{
public:
    ONNXModel();

    void load(const std::string& path) override;

    std::vector<float> predict(const std::string& input_text) override;

    const std::vector<const char*>& getInputNames(){ return input_names_; }

    const std::vector<const char*>& getOutputNames(){ return output_names_; }
private:
    Ort::Env env_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> session_;
    
    std::vector<std::string> input_names_storage_;
    std::vector<std::string> output_names_storage_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};