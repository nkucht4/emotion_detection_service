add_executable(EmotionDetectionService main.cpp)

target_include_directories(EmotionDetectionService PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../generated
)

target_link_libraries(EmotionDetectionService PRIVATE
    service      
    gRPC::grpc++
    protobuf::libprotobuf
    pthread
)