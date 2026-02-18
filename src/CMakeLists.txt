cmake_minimum_required(VERSION 3.22)
project(EmotionDetectionService LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(utf8_range CONFIG REQUIRED)

# Proto and generated output
set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/protos")
set(GENERATED_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")

add_subdirectory(src/nlp)          
add_subdirectory(src/service)  
add_subdirectory(src/app)           