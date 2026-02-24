#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include "service/EmotionService.hpp"
#include "emotion_service.pb.h"

class EmotionServiceTest : public testing::Test{
protected:
    std::unique_ptr<EmotionServiceImpl> service;

    void SetUp() override{
        service = std::make_unique<EmotionServiceImpl>();
    }

    grpc::ServerContext context;
};

TEST_F(EmotionServiceTest, ReturnsEmotionScoresForValidText){
    Text request;
    request.set_text("I am very happy today");

    EmotionResponse response;

    grpc::Status status =
        service->GetTextEmotion(&context, &request, &response);

    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(response.scores().empty());
}

TEST_F(EmotionServiceTest, ReturnsInvalidArgumentWhenRequestIsNull){
    EmotionResponse response;

    grpc::Status status =
        service->GetTextEmotion(&context, nullptr, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(EmotionServiceTest, HandlesEmptyText){
    Text request;
    request.set_text("");

    EmotionResponse response;

    grpc::Status status =
        service->GetTextEmotion(&context, &request, &response);

    EXPECT_TRUE(status.ok());
}

