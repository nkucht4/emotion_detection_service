add_library(service STATIC
    # sources
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
)
