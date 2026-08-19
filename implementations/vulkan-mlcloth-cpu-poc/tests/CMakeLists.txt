cmake_minimum_required(VERSION 3.15)
project(mlcloth_format_tests LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /WX /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(format_tests
    ../include/mlcloth_formats.cpp
    format_tests.cpp
)

target_include_directories(format_tests PRIVATE ../include)
add_test(NAME mlcloth_formats COMMAND format_tests)

if(WIN32)
    add_executable(runtime_integration
        ../include/mlcloth_formats.cpp
        ../include/mlcloth_runtime.cpp
        runtime_integration.cpp
    )
    target_include_directories(runtime_integration PRIVATE ../include)
    set(DEFAULT_RUNTIME "${CMAKE_CURRENT_SOURCE_DIR}/../.work/runtime")
    set(DEFAULT_MODEL "${DEFAULT_RUNTIME}/model_NeuralRes4_NeuralRes4_final.enc")
    set(DEFAULT_CLIP "${CMAKE_CURRENT_SOURCE_DIR}/../.work/clips/AS_C10032_ArmedSprint_Skirt.mldrv")
    if(EXISTS "${DEFAULT_RUNTIME}/AILab.dll" AND EXISTS "${DEFAULT_MODEL}" AND EXISTS "${DEFAULT_CLIP}")
        add_test(NAME mlcloth_ailab_runtime COMMAND runtime_integration "${DEFAULT_RUNTIME}" "${DEFAULT_MODEL}" "${DEFAULT_CLIP}")
    else()
        message(STATUS "AILab integration test not registered: run prepare_runtime.ps1 and bake_driver_clip.ps1 first")
    endif()
endif()
