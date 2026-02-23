#include <gtest/gtest.h>
#include <iostream>
#include "nlp/model/ONNXModel.hpp"
#include "nlp/utils/LabelMapper.hpp"

class ModelLabelMapperIntegrationTest : public testing::Test{
protected:
    std::unique_ptr<ONNXModel> model;
    std::unique_ptr<LabelMapper> mapper;

    void SetUp() override{
        model = std::make_unique<ONNXModel>();
        model->load(TEST_MODEL_PATH);

        std::vector<std::string> emotions = {
            "admiration", "amusement", "anger", "annoyance", "approval", 
            "caring", "confusion", "curiosity", "desire", "disappointment", 
            "disapproval", "disgust", "embarrassment", "excitement", "fear", 
            "gratitude", "grief", "joy", "love", "nervousness", "optimism", 
            "pride", "realization", "relief", "remorse", "sadness", "surprise",
            "neutral"
        };

        mapper = std::make_unique<LabelMapper>(emotions);
    }
};

TEST_F(ModelLabelMapperIntegrationTest, PredictAndMapLabels){
    Eigen::VectorXf input_vec = Eigen::VectorXf::Random(20000);

    std::vector<float> model_output = model->predict(input_vec);

    ASSERT_FALSE(model_output.empty());

    auto mapped_labels = mapper->map(model_output);

    ASSERT_EQ(mapped_labels.size(), model_output.size());

    for (const auto& [label, score] : mapped_labels) {
        std::cout << label << ": " << score << std::endl;
    }
}

TEST_F(ModelLabelMapperIntegrationTest, DeterministicMapping){
    Eigen::VectorXf input_vec = Eigen::VectorXf::Random(20000);

    std::vector<float> output1 = model->predict(input_vec);
    std::vector<float> output2 = model->predict(input_vec);

    auto labels1 = mapper->map(output1);
    auto labels2 = mapper->map(output2);

    ASSERT_EQ(labels1.size(), labels2.size());

    std::vector<std::pair<std::string, float>> vec1(labels1.begin(), labels1.end());
    std::vector<std::pair<std::string, float>> vec2(labels2.begin(), labels2.end());

    auto cmp = [](const auto &a, const auto &b){ return a.first < b.first; };
    std::sort(vec1.begin(), vec1.end(), cmp);
    std::sort(vec2.begin(), vec2.end(), cmp);

    for (size_t i = 0; i < vec1.size(); ++i)
        EXPECT_FLOAT_EQ(vec1[i].second, vec2[i].second);
}