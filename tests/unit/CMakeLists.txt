cmake_minimum_required(VERSION 3.14)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(test_preprocessor test_preprocessor.cpp)
target_link_libraries(test_preprocessor PRIVATE nlp_module gtest_main)

include(GoogleTest)
gtest_discover_tests(test_preprocessor
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)