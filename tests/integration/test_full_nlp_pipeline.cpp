#include <gtest/gtest.h>
#include "nlp/pipeline/NLPPipeline.hpp"

class FullNLPPipelineTest : public testing::Test{
protected:
    std::unique_ptr<NLPPipeline> pipeline;
    std::vector<std::string> emotions;

    void SetUp() override{
        emotions = {
            "admiration", "amusement", "anger", "annoyance", "approval", 
            "caring", "confusion", "curiosity", "desire", "disappointment", 
            "disapproval", "disgust", "embarrassment", "excitement", "fear", 
            "gratitude", "grief", "joy", "love", "nervousness", "optimism", 
            "pride", "realization", "relief", "remorse", "sadness", "surprise",
            "neutral"
        };

        pipeline = std::make_unique<NLPPipeline>(
            TEST_VOCAB_PATH, 
            TEST_IDF_PATH,
            TEST_MODEL_PATH, 
            emotions);
    }
};

TEST_F(FullNLPPipelineTest, ProcessAndPredict){
    std::string input = "I am very happy today!";

    auto result = pipeline->run(input);

    EXPECT_EQ(result.size(), emotions.size());

    for (const auto& label : emotions){
        EXPECT_TRUE(result.find(label) != result.end());
    }
}

TEST_F(FullNLPPipelineTest, DeterministicOutput){
    auto result1 = pipeline->run("This is a test sentence.");
    auto result2 = pipeline->run("This is a test sentence.");

    ASSERT_EQ(result1.size(), result2.size());

    for (const auto& [label, value] : result1) {
        EXPECT_NEAR(value, result2[label], 1e-6f);
    }
}

TEST_F(FullNLPPipelineTest, HandlesEmptyInput){
    EXPECT_THROW(pipeline->run(""), std::invalid_argument);
}

TEST_F(FullNLPPipelineTest, NoNegativeProbabilities){
    auto result = pipeline->run("I feel terrible.");

    for (const auto& [label, value] : result) {
        EXPECT_GE(value, 0.0f);
        EXPECT_LE(value, 1.0f);
    }
}