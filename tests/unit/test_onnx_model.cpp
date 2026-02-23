#include <gtest/gtest.h>
#include "nlp/model/ONNXModel.hpp"

class ONNXModelTest : public testing::Test{
protected:
    ONNXModel model;

    void SetUp() override {
        model.load(TEST_MODEL_PATH);
    }
};

TEST_F(ONNXModelTest, LoadModel){
    EXPECT_FALSE(model.getInputNames().empty());
    EXPECT_FALSE(model.getOutputNames().empty());
}

TEST_F(ONNXModelTest, PredictSingleInput){
    Eigen::VectorXf input(20000); 
    input.setRandom();

    std::vector<float> output = model.predict(input);
    EXPECT_FALSE(output.empty());
}

TEST_F(ONNXModelTest, EmptyInput){
    Eigen::VectorXf input(20000);
    input.setZero();
    std::vector<float> output = model.predict(input);
    EXPECT_FALSE(output.empty());
}