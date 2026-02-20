cmake_minimum_required(VERSION 3.14)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_classical_preprocessor test_classical_preprocessor.cpp)
target_link_libraries(test_classical_preprocessor PRIVATE nlp_module gtest_main)

include(GoogleTest)
gtest_discover_tests(test_classical_preprocessor
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)