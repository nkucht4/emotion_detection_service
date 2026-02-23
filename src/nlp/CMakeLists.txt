set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.24.2)
set(EIGEN3_DIR ${CMAKE_SOURCE_DIR}/eigen)

add_library(onnxruntime SHARED IMPORTED)

set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_DIR}/include
)

add_library(nlp_module STATIC
    preprocessing/ClassicalPreprocessor.cpp
    model/ONNXModel.cpp
)

target_include_directories(nlp_module PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${EIGEN3_DIR}
)

target_link_libraries(nlp_module PUBLIC
    utf8_range::utf8_range   
    protobuf::libprotobuf    
    onnxruntime
)
