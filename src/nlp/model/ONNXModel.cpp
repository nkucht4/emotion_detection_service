#include "nlp/model/ONNXModel.hpp"

ONNXModel::ONNXModel() : env_(ORT_LOGGING_LEVEL_WARNING, "ONNXModel"){}

void ONNXModel::load(const std::string& model_path){
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

    input_names_ = session_->GetInputNames();
    output_names_ = session_->GetOutputNames();
}

std::vector<float> ONNXModel::predict(const std::string& input_text){
    std::vector<const char*> input_strs = { input_text.c_str() };
    std::vector<int64_t> input_shape = {1};

    Ort::Value input_tensor = Ort::Value::CreateTensor(
        allocator_, 
        input_shape.data(), 
        input_shape.size(), 
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
    );

    Ort::GetApi().FillStringTensor(
        input_tensor, 
        input_strs.data(), 
        input_strs.size()
    );

    std::vector<Ort::Value> output_tensors(output_names_.size());

    session_->Run(
        Ort::RunOptions{nullptr},         
        input_names_.data(),              
        &input_tensor,                   
        1,                          
        output_names_.data(),       
        output_tensors.data(),  
        output_tensors.size()
    );

    float* float_array = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    std::vector<float> output(float_array, float_array + output_size);
}