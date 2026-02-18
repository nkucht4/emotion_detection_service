#include "service/emotion_service.hpp"

EmotionServiceImpl::EmotionServiceImpl(){
    // TO DO
}

grpc::Status EmotionServiceImpl::GetTextEmotion(grpc::ServerContext* context, const Text* request, EmotionResponse* response){
    // TO DO
    return grpc::Status::IgnoreError;
}