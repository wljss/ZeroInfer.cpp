cmake_minimum_required(VERSION 3.16)

project(ZeroInfer LANGUAGES CXX)

include(CTest)

find_package(OpenMP REQUIRED)

add_library(zeroinfer STATIC
    src/internal/ops.cpp
    src/zeroinfer.cpp
)

target_compile_features(zeroinfer PUBLIC cxx_std_17)
target_include_directories(zeroinfer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(zeroinfer PUBLIC OpenMP::OpenMP_CXX)

add_executable(llama_infer
    main.cpp
)

target_link_libraries(llama_infer PRIVATE zeroinfer)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(zeroinfer PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(llama_infer PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
    add_executable(zeroinfer_ops_tests
        tests/test_ops.cpp
    )

    target_include_directories(zeroinfer_ops_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(zeroinfer_ops_tests PRIVATE zeroinfer)

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(zeroinfer_ops_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME zeroinfer.ops.matmul COMMAND zeroinfer_ops_tests matmul)
    add_test(NAME zeroinfer.ops.rmsnorm COMMAND zeroinfer_ops_tests rmsnorm)
    add_test(NAME zeroinfer.ops.softmax COMMAND zeroinfer_ops_tests softmax)
endif()
