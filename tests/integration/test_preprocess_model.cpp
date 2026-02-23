#include <gtest/gtest.h>
#include <iostream>
#include "nlp/preprocessing/ClassicalPreprocessor.hpp"
#include "nlp/model/ONNXModel.hpp"

class PreprocessModelIntegrationTest : public testing::Test{
protected:
    ClassicalPreprocessor prep;
    std::unique_ptr<ONNXModel> model;

    void SetUp() override{
        model = std::make_unique<ONNXModel>();
        model->load(TEST_MODEL_PATH);
    }
};

TEST_F(PreprocessModelIntegrationTest, ProcessTextAndReturnPrediction){
    std::string input = "I am very happy today!";

    auto prep_input = prep.preprocess(input);

    ASSERT_FALSE(prep_input.empty());

    auto output = model->predict(prep_input);

    ASSERT_FALSE(output.empty());   
}

TEST_F(PreprocessModelIntegrationTest, DeterministicOutput){
    std::string input = "I love NLP";

    auto output1 = model->predict(prep.preprocess(input));
    auto output2 = model->predict(prep.preprocess(input));

    ASSERT_EQ(output1.size(), output2.size());
    for (size_t i = 0; i < output1.size(); ++i)
        EXPECT_FLOAT_EQ(output1[i], output2[i]);

}