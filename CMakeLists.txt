cmake_minimum_required(VERSION 3.16)

project(ZeroInfer LANGUAGES CXX)

include(CTest)

option(
    ZEROINFER_ENABLE_PYTORCH_ALIGNMENT
    "Enable optional operator alignment tests against PyTorch"
    OFF
)

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

    if(ZEROINFER_ENABLE_PYTORCH_ALIGNMENT)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)

        execute_process(
            COMMAND
                ${Python3_EXECUTABLE}
                -c
                "import numpy, torch"
            RESULT_VARIABLE ZEROINFER_PYTORCH_IMPORT_RESULT
            ERROR_VARIABLE ZEROINFER_PYTORCH_IMPORT_ERROR
        )
        if(NOT ZEROINFER_PYTORCH_IMPORT_RESULT EQUAL 0)
            message(
                FATAL_ERROR
                "Python interpreter cannot import NumPy and PyTorch:\n"
                "${ZEROINFER_PYTORCH_IMPORT_ERROR}"
            )
        endif()

        add_executable(zeroinfer_ops_runner
            tests/pytorch/ops_runner.cpp
        )
        target_include_directories(zeroinfer_ops_runner
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(zeroinfer_ops_runner PRIVATE zeroinfer)

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(
                zeroinfer_ops_runner
                PRIVATE
                    -Wall
                    -Wextra
                    -Wpedantic
            )
        endif()

        add_test(
            NAME zeroinfer.pytorch.ops
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/pytorch/align_ops.py
                --runner
                $<TARGET_FILE:zeroinfer_ops_runner>
        )
        set_tests_properties(
            zeroinfer.pytorch.ops
            PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=1"
                TIMEOUT 60
        )
    endif()
endif()
