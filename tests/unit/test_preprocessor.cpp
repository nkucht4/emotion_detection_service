#include <gtest/gtest.h>
#include "nlp/preprocessing/Preprocessor.hpp"

class PreprocessorTest : public testing::Test{
protected:
    Preprocessor prep;
};

TEST_F(PreprocessorTest, FullPipelineTest){
    std::string text = "Hello WORLD, what a nice day!";

    auto tokens = prep.preprocessFully(text);

    std::vector<std::string> expected = {"hello", "world", "what", "nice", "day!"};
    EXPECT_EQ(tokens, expected);
}

TEST_F(PreprocessorTest, HandlesEmptyString){
    std::string text = "";
    auto tokens = prep.preprocessFully(text);

    EXPECT_TRUE(tokens.empty());
}

TEST_F(PreprocessorTest, OnlyWhitespace){
    auto tokens = prep.preprocessFully("     ");
    EXPECT_TRUE(tokens.empty());
}

TEST_F(PreprocessorTest, OnlyPunctuation) {
    auto tokens = prep.preprocessFully(",,,...");
    EXPECT_TRUE(tokens.empty());
}

TEST_F(PreprocessorTest, CaseNormalization){
    auto tokens = prep.preprocessFully("HeLLo WoRLD");
    std::vector<std::string> expected = {"hello", "world"};
    EXPECT_EQ(tokens, expected);
}

TEST_F(PreprocessorTest, StopwordsRemoved){
    auto tokens = prep.preprocessFully("I am the batman");
    std::vector<std::string> expected = {"batman"};
    EXPECT_EQ(tokens, expected);
}

TEST_F(PreprocessorTest, StemmingRemovesSuffix){
    auto tokens = prep.preprocessFully("running jumped fastest");
    std::vector<std::string> expected = {"runn", "jump", "fast"};
    EXPECT_EQ(tokens, expected);
}

TEST_F(PreprocessorTest, DoesNotOverStemShortWords){
    auto tokens = prep.preprocessFully("is as us");
    std::vector<std::string> expected = {"as"};
    EXPECT_EQ(tokens, expected); 
}

TEST_F(PreprocessorTest, ShortWordsRemainStable){
    auto tokens = prep.preprocessFully("gas");
    std::vector<std::string> expected = {"gas"};
    EXPECT_EQ(tokens, expected);
}

TEST_F(PreprocessorTest, HandlesMultipleSpaces){
    auto tokens = prep.preprocessFully("hello     world");
    std::vector<std::string> expected = {"hello", "world"};
    EXPECT_EQ(tokens, expected);
}

TEST_F(PreprocessorTest, ComplexSentence){
    auto tokens = prep.preprocessFully(
        "I was running quickly, and I absolutely loved it!"
    );

    std::vector<std::string> expected = {
        "runn", "quick", "absolute", "lov", "it!"
    };

    EXPECT_EQ(tokens, expected);
}