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

    const std::vector<std::string>& getInputNames(){ return input_names_; }

    const std::vector<std::string>& getOutputNames(){ return output_names_; }
private:
    Ort::Env env_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};