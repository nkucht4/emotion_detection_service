#include "nlp/model/ONNXModel.hpp"

ONNXModel::ONNXModel() : env_(ORT_LOGGING_LEVEL_WARNING, "ONNXModel"){}

void ONNXModel::load(const std::string& model_path){
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

    auto input_names_str = session_->GetInputNames();
    auto output_names_str = session_->GetOutputNames();

    input_names_storage_.clear();
    output_names_storage_.clear();
    for (const auto& name : input_names_str) input_names_storage_.push_back(name);
    for (const auto& name : output_names_str) output_names_storage_.push_back(name);

    input_names_.clear();
    for (const auto& s : input_names_storage_) input_names_.push_back(s.c_str());

    output_names_.clear();
    for (const auto& s : output_names_storage_) output_names_.push_back(s.c_str());
}

std::vector<float> ONNXModel::predict(const std::string& input_text){
    std::vector<const char*> input_strs = { input_text.c_str() };
    std::vector<int64_t> input_shape = {1, 1};

    Ort::Value input_tensor = Ort::Value::CreateTensor(
        allocator_,
        input_shape.data(),
        input_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
    );

    const char* input_str = input_text.c_str();
    Ort::GetApi().FillStringTensor(input_tensor, &input_str, 1);

    std::vector<Ort::Value> output_tensors(output_names_.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));

    session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names_.data(),
        output_tensors.data(),
        output_tensors.size()
    );

    Ort::Value& output_tensor = output_tensors[0];
    float* float_array = output_tensor.GetTensorMutableData<float>();
    size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    return std::vector<float>(float_array, float_array + output_size);
}

std::vector<float> ONNXModel::predict(const Eigen::VectorXf& input_vec){
    std::vector<int64_t> shape = {1, static_cast<int64_t>(input_vec.size())};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator, OrtMemTypeCPU
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(input_vec.data()),
        input_vec.size(),
        shape.data(),
        shape.size()
    );

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));

    std::vector<Ort::Value> output_tensors(output_names_.size());

    session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names_.data(),
        output_tensors.data(),
        output_tensors.size()
    );

    Ort::Value& output_tensor = output_tensors[0];
    float* float_array = output_tensor.GetTensorMutableData<float>();
    size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    return std::vector<float>(float_array, float_array + output_size);
}