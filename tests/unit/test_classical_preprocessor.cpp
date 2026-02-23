#include <gtest/gtest.h>
#include "nlp/preprocessing/ClassicalPreprocessor.hpp"

class ClassicalPreprocessorTest : public testing::Test{
protected:
    std::unique_ptr<ClassicalPreprocessor> prep;

    void SetUp() override{
        prep = std::make_unique<ClassicalPreprocessor>(
            TEST_VOCAB_PATH,
            TEST_IDF_PATH
        );
    }
};

TEST_F(ClassicalPreprocessorTest, FullPipelineTest){
    std::string text = "Hello WORLD, what a nice day!";

    auto output = prep->preprocessToString(text);

    std::string expected = {"hello world what nice day!"};
    EXPECT_EQ(output, expected);
}

TEST_F(ClassicalPreprocessorTest, HandlesEmptyString){
    std::string text = "";
    auto output = prep->preprocessToString(text);

    EXPECT_TRUE(output.empty());
}

TEST_F(ClassicalPreprocessorTest, OnlyWhitespace){
    auto output = prep->preprocessToString("     ");
    EXPECT_TRUE(output.empty());
}

TEST_F(ClassicalPreprocessorTest, OnlyPunctuation) {
    auto output = prep->preprocessToString(",,,...");
    EXPECT_TRUE(output.empty());
}

TEST_F(ClassicalPreprocessorTest, CaseNormalization){
    auto output = prep->preprocessToString("HeLLo WoRLD");
    std::string expected = {"hello world"};
    EXPECT_EQ(output, expected);
}

TEST_F(ClassicalPreprocessorTest, StopwordsRemoved){
    auto output = prep->preprocessToString("I am the batman");
    std::string expected = {"batman"};
    EXPECT_EQ(output, expected);
}

TEST_F(ClassicalPreprocessorTest, StemmingRemovesSuffix){
    auto output = prep->preprocessToString("running jumped fastest");
    std::string expected = {"runn jump fast"};
    EXPECT_EQ(output, expected);
}

TEST_F(ClassicalPreprocessorTest, DoesNotOverStemShortWords){
    auto output = prep->preprocessToString("is as us");
    std::string expected = {"as"};
    EXPECT_EQ(output, expected); 
}

TEST_F(ClassicalPreprocessorTest, ShortWordsRemainStable){
    auto output = prep->preprocessToString("gas");
    std::string expected = {"gas"};
    EXPECT_EQ(output, expected);
}

TEST_F(ClassicalPreprocessorTest, HandlesMultipleSpaces){
    auto output = prep->preprocessToString("hello     world");
    std::string expected = {"hello world"};
    EXPECT_EQ(output, expected);
}

TEST_F(ClassicalPreprocessorTest, ComplexSentence){
    auto output = prep->preprocessToString(
        "I was running quickly, and I absolutely loved it!"
    );

    std::string expected = { "runn quick absolute lov it!" };

    EXPECT_EQ(output, expected);
}

TEST_F(ClassicalPreprocessorTest, Vector_EmptyInputReturnsZeroVector){
    Eigen::VectorXf vec = prep->preprocessToVector("");

    EXPECT_EQ(vec.size(), prep->preprocessToVector("test").size());
    EXPECT_TRUE(vec.isZero());
}

TEST_F(ClassicalPreprocessorTest, Vector_WhitespaceReturnsZeroVector){
    Eigen::VectorXf vec = prep->preprocessToVector("     ");

    EXPECT_TRUE(vec.isZero());
}

TEST_F(ClassicalPreprocessorTest, Vector_CaseNormalizationConsistent){
    Eigen::VectorXf a = prep->preprocessToVector("HeLLo WoRLD");
    Eigen::VectorXf b = prep->preprocessToVector("hello world");

    EXPECT_TRUE(a.isApprox(b));
}

TEST_F(ClassicalPreprocessorTest, Vector_StopwordsRemovedEquivalent){
    Eigen::VectorXf a = prep->preprocessToVector("I am the batman");
    Eigen::VectorXf b = prep->preprocessToVector("batman");

    EXPECT_TRUE(a.isApprox(b));
}

TEST_F(ClassicalPreprocessorTest, Vector_StemmingConsistent){
    Eigen::VectorXf a = prep->preprocessToVector("running jumped fastest");
    Eigen::VectorXf b = prep->preprocessToVector("runn jump fast");

    EXPECT_TRUE(a.isApprox(b));
}

TEST_F(ClassicalPreprocessorTest, Vector_MultipleSpacesEquivalent){
    Eigen::VectorXf a = prep->preprocessToVector("hello     world");
    Eigen::VectorXf b = prep->preprocessToVector("hello world");

    EXPECT_TRUE(a.isApprox(b));
}