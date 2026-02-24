set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(nlohmann_json 
URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(nlohmann_json)

find_package(Protobuf REQUIRED)
message(STATUS "Protobuf version: ${Protobuf_VERSION}")
message(STATUS "Protoc executable: ${Protobuf_PROTOC_EXECUTABLE}")
find_package(gRPC CONFIG REQUIRED)
find_package(utf8_range CONFIG REQUIRED)

# Proto and generated output
set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/protos")
set(GENERATED_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")

add_subdirectory(nlp)
add_subdirectory(service)
#add_subdirectory(app)