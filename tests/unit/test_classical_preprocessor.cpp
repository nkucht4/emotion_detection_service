#include <gtest/gtest.h>
#include "nlp/preprocessing/ClassicalPreprocessor.hpp"

class ClassicalPreprocessorTest : public testing::Test{
protected:
    ClassicalPreprocessor prep;
};

TEST_F(ClassicalPreprocessorTest, FullPipelineTest){
    std::string text = "Hello WORLD, what a nice day!";

    auto output = prep.preprocess(text);

    std::string expected = {"hello world what nice day!"};
    EXPECT_EQ(output, expected);
}

TEST_F(ClassicalPreprocessorTest, HandlesEmptyString){
    std::string text = "";
    auto output = prep.preprocess(text);

    EXPECT_TRUE(output.empty());
}

TEST_F(ClassicalPreprocessorTest, OnlyWhitespace){
    auto output = prep.preprocess("     ");
    EXPECT_TRUE(output.empty());
}

TEST_F(ClassicalPreprocessorTest, OnlyPunctuation) {
    auto output = prep.preprocess(",,,...");
    EXPECT_TRUE(output.empty());
}

TEST_F(ClassicalPreprocessorTest, CaseNormalization){
    auto output = prep.preprocess("HeLLo WoRLD");
    std::string expected = {"hello world"};
    EXPECT_EQ(output, expected);
}

TEST_F(ClassicalPreprocessorTest, StopwordsRemoved){
    auto output = prep.preprocess("I am the batman");
    std::string expected = {"batman"};
    EXPECT_EQ(output, expected);
}

TEST_F(ClassicalPreprocessorTest, StemmingRemovesSuffix){
    auto output = prep.preprocess("running jumped fastest");
    std::string expected = {"runn jump fast"};
    EXPECT_EQ(output, expected);
}

TEST_F(ClassicalPreprocessorTest, DoesNotOverStemShortWords){
    auto output = prep.preprocess("is as us");
    std::string expected = {"as"};
    EXPECT_EQ(output, expected); 
}

TEST_F(ClassicalPreprocessorTest, ShortWordsRemainStable){
    auto output = prep.preprocess("gas");
    std::string expected = {"gas"};
    EXPECT_EQ(output, expected);
}

TEST_F(ClassicalPreprocessorTest, HandlesMultipleSpaces){
    auto output = prep.preprocess("hello     world");
    std::string expected = {"hello world"};
    EXPECT_EQ(output, expected);
}

TEST_F(ClassicalPreprocessorTest, ComplexSentence){
    auto output = prep.preprocess(
        "I was running quickly, and I absolutely loved it!"
    );

    std::string expected = { "runn quick absolute lov it!" };

    EXPECT_EQ(output, expected);
}