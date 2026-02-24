cmake_minimum_required(VERSION 3.14)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_integration test_preprocess_model.cpp test_model_mapper.cpp 
    test_full_nlp_pipeline.cpp )
target_link_libraries(test_integration PRIVATE nlp_module gtest_main)

target_compile_definitions(test_integration
    PRIVATE TEST_MODEL_PATH="${CMAKE_SOURCE_DIR}/ml/onnx_test_model/emotion_classifier_float.onnx"
    TEST_VOCAB_PATH="${CMAKE_SOURCE_DIR}/ml/onnx_test_model/vocab.txt"
    TEST_IDF_PATH="${CMAKE_SOURCE_DIR}/ml/onnx_test_model/idf.txt"
)

include(GoogleTest)
gtest_discover_tests(test_integration
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)