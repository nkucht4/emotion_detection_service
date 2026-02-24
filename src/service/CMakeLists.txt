add_library(service STATIC
    EmotionService.cpp
)

target_include_directories(service PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include      
    ${CMAKE_CURRENT_SOURCE_DIR}/../generated   
)

target_link_libraries(service PUBLIC
    nlp_module              
    gRPC::grpc++            
    protobuf::libprotobuf
    pthread
    PRIVATE
        nlohmann_json::nlohmann_json
)
