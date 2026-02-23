#include <gtest/gtest.h>
#include "nlp/utils/LabelMapper.hpp"

class LabelMapperTest : public testing::Test{
protected:
    std::unique_ptr<LabelMapper> prep;

    void SetUp() override{
        prep = std::make_unique<LabelMapper>(
            std::vector<std::string>{"joy", "sadness", "anger"}
        );
    }
};

TEST_F(LabelMapperTest, MapsCorrectly){
    std::vector<float> scores = {0.1f, 0.7f, 0.2f};
    auto mapped = prep->map(scores);

    EXPECT_FLOAT_EQ(mapped["joy"], 0.1f);
    EXPECT_FLOAT_EQ(mapped["sadness"], 0.7f);
    EXPECT_FLOAT_EQ(mapped["anger"], 0.2f);
}

TEST_F(LabelMapperTest, ThrowsOnSizeMismatch){
    std::vector<float> scores = {0.1f, 0.7f, 0.2f, 0.5f};
    EXPECT_THROW(prep->map(scores), std::runtime_error);
}