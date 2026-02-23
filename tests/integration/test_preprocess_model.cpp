#include <gtest/gtest.h>
#include <iostream>
#include "nlp/preprocessing/ClassicalPreprocessor.hpp"
#include "nlp/model/ONNXModel.hpp"

class PreprocessModelIntegrationTest : public testing::Test{
protected:
    std::unique_ptr<ClassicalPreprocessor> prep;
    std::unique_ptr<ONNXModel> model;

    void SetUp() override{
        prep = std::make_unique<ClassicalPreprocessor>(TEST_VOCAB_PATH, TEST_IDF_PATH);
        model = std::make_unique<ONNXModel>();
        model->load(TEST_MODEL_PATH);
    }
};

TEST_F(PreprocessModelIntegrationTest, ProcessTextAndReturnPrediction){
    std::string input = "I am very happy today!";

    Eigen::VectorXf prep_input = prep->preprocessToVector(input);

    ASSERT_FALSE(prep_input.size() == 0);

    auto output = model->predict(prep_input);

    ASSERT_FALSE(output.empty());
}

TEST_F(PreprocessModelIntegrationTest, DeterministicOutput){
    std::string input = "I love NLP";

    std::vector<float> output1 = model->predict(prep->preprocessToVector(input));
    std::vector<float> output2 = model->predict(prep->preprocessToVector(input));

    ASSERT_EQ(output1.size(), output2.size());

    for (size_t i = 0; i < output1.size(); ++i)
        EXPECT_FLOAT_EQ(output1[i], output2[i]);
}