#pragma once

#include <grpcpp/grpcpp.h>

#include "emotion_service.grpc.pb.h"
#include "emotion_service.pb.h"
#include "nlp/pipeline/NLPPipeline.hpp"

class EmotionServiceImpl final : public EmotionService::Service {
public:
    EmotionServiceImpl();

    grpc::Status GetTextEmotion(grpc::ServerContext* context, const Text* request, EmotionResponse* response) override;

private:
    std::unique_ptr<NLPPipeline> pipeline;
};