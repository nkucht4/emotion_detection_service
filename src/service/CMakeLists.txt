cmake_minimum_required(VERSION 3.16)

add_library(service STATIC
    # sources
)

target_include_directories(grpc_service PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include      
    ${CMAKE_CURRENT_SOURCE_DIR}/../generated   
)

target_link_libraries(grpc_service PUBLIC
    nlp_module              
    gRPC::grpc++            
    protobuf::libprotobuf
    pthread
)
