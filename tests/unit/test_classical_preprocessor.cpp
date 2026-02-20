#include <gtest/gtest.h>
#include "nlp/preprocessing/ClassicalPreprocessor.hpp"

class ClassicalPreprocessorTest : public testing::Test{
protected:
    ClassicalPreprocessor prep;
};

TEST_F(ClassicalPreprocessorTest, FullPipelineTest){
    std::string text = "Hello WORLD, what a nice day!";

    auto tokens = prep.preprocess(text);

    std::vector<std::string> expected = {"hello", "world", "what", "nice", "day!"};
    EXPECT_EQ(tokens, expected);
}

TEST_F(ClassicalPreprocessorTest, HandlesEmptyString){
    std::string text = "";
    auto tokens = prep.preprocess(text);

    EXPECT_TRUE(tokens.empty());
}

TEST_F(ClassicalPreprocessorTest, OnlyWhitespace){
    auto tokens = prep.preprocess("     ");
    EXPECT_TRUE(tokens.empty());
}

TEST_F(ClassicalPreprocessorTest, OnlyPunctuation) {
    auto tokens = prep.preprocess(",,,...");
    EXPECT_TRUE(tokens.empty());
}

TEST_F(ClassicalPreprocessorTest, CaseNormalization){
    auto tokens = prep.preprocess("HeLLo WoRLD");
    std::vector<std::string> expected = {"hello", "world"};
    EXPECT_EQ(tokens, expected);
}

TEST_F(ClassicalPreprocessorTest, StopwordsRemoved){
    auto tokens = prep.preprocess("I am the batman");
    std::vector<std::string> expected = {"batman"};
    EXPECT_EQ(tokens, expected);
}

TEST_F(ClassicalPreprocessorTest, StemmingRemovesSuffix){
    auto tokens = prep.preprocess("running jumped fastest");
    std::vector<std::string> expected = {"runn", "jump", "fast"};
    EXPECT_EQ(tokens, expected);
}

TEST_F(ClassicalPreprocessorTest, DoesNotOverStemShortWords){
    auto tokens = prep.preprocess("is as us");
    std::vector<std::string> expected = {"as"};
    EXPECT_EQ(tokens, expected); 
}

TEST_F(ClassicalPreprocessorTest, ShortWordsRemainStable){
    auto tokens = prep.preprocess("gas");
    std::vector<std::string> expected = {"gas"};
    EXPECT_EQ(tokens, expected);
}

TEST_F(ClassicalPreprocessorTest, HandlesMultipleSpaces){
    auto tokens = prep.preprocess("hello     world");
    std::vector<std::string> expected = {"hello", "world"};
    EXPECT_EQ(tokens, expected);
}

TEST_F(ClassicalPreprocessorTest, ComplexSentence){
    auto tokens = prep.preprocess(
        "I was running quickly, and I absolutely loved it!"
    );

    std::vector<std::string> expected = {
        "runn", "quick", "absolute", "lov", "it!"
    };

    EXPECT_EQ(tokens, expected);
}