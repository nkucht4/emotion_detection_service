cmake_minimum_required(VERSION 3.14)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_unit test_classical_preprocessor.cpp test_onnx_model.cpp)
target_link_libraries(test_unit PRIVATE nlp_module gtest_main)

target_compile_definitions(test_unit
    PRIVATE TEST_MODEL_PATH="${CMAKE_SOURCE_DIR}/ml/onnx_test_model/emotion_classifier.onnx"
)

include(GoogleTest)
gtest_discover_tests(test_unit
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)