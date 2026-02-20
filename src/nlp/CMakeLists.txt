add_library(nlp_module STATIC
    preprocessing/ClassicalPreprocessor.cpp
)

target_include_directories(nlp_module PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(nlp_module PUBLIC
    utf8_range::utf8_range   
    protobuf::libprotobuf    
)
