#include "service/EmotionService.hpp"
#include "utils/ConfigReader.hpp"
#include "emotion_service.pb.h"

EmotionServiceImpl::EmotionServiceImpl(){
    NLPConfig cfg = ConfigReader::readNLPConfig("config/config.json");
    pipeline = std::make_unique<NLPPipeline>(
        cfg.vocab_path,
        cfg.idf_path,
        cfg.model_path,
        cfg.labels
    );
}

grpc::Status EmotionServiceImpl::GetTextEmotion(grpc::ServerContext* context, const Text* request, EmotionResponse* response){
    if (!pipeline){
        return grpc::Status(grpc::StatusCode::INTERNAL, "Pipeline not initialized");
    }

    if (!request) {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            "Request is null"
        );
    }

    const std::string& text = request->text();

    auto results = pipeline->run(text);

    for (const auto& [label, score] : results){
        (*response->mutable_scores())[label] = score;
    }

    return grpc::Status::OK;
}