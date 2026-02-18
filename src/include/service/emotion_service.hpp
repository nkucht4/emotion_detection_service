#pragma once

#include <grpcpp/grpcpp.h>

#include "emotion_service.grpc.pb.h"
#include "emotion_service.pb.h"

class EmotionServiceImpl final : public EmotionService::Service {
public:
    EmotionServiceImpl();

    grpc::Status GetTextEmotion(grpc::ServerContext* context, const Text* request, EmotionResponse* response) override;
};