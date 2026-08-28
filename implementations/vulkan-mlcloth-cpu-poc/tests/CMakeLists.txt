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

# Strict validator for a baked mesh, and the cross-check between the C++ and Python
# derivations of the same topology contract. Useful standalone before running the
# sample, which is why it is a tool rather than only a test body.
add_executable(mesh_validate
    ../include/mlcloth_formats.cpp
    mesh_validate.cpp
)
target_include_directories(mesh_validate PRIVATE ../include)

# Runs one XPBD scenario so the Python reference in the sibling PoC can be compared
# against this port. The comparison is the only check here that does not share the port's
# own assumptions, so it is a first-class artifact rather than a test detail.
add_executable(xpbd_reference
    ../include/mlcloth_xpbd.cpp
    xpbd_reference.cpp
)
target_include_directories(xpbd_reference PRIVATE ../include)

# The Python side: synthesises MLCloth training-export fixtures, runs the converter
# and re-checks the container against the same rules parse_mesh enforces. It shells
# out to `mesh_validate` when it is available, so a disagreement between the two
# implementations fails the build rather than surfacing later as a scrambled mesh.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME mlcloth_mesh_converter
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_mlcloth_mesh.py")
    set_tests_properties(mlcloth_mesh_converter PROPERTIES
        ENVIRONMENT "MLCLOTH_MESH_VALIDATE=$<TARGET_FILE:mesh_validate>")
    # Unreal's rotator convention, the capsule signed distance, and the real physics
    # asset. Needs no built binary, so it runs even on a fresh checkout.
    add_test(NAME mlcloth_capsules
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_mlcloth_capsules.py")
    # The offline driver bake. Its most important assertion compares against a clip
    # Unreal produced, which is skipped with a printed note when those assets are absent.
    add_test(NAME mlcloth_drivers
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_mlcloth_drivers.py")
    # The CPU XPBD port against the Python reference it was ported from. Needs the sibling
    # PoC's virtual environment for torch; prints a skip note when it is absent.
    add_test(NAME mlcloth_xpbd
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_mlcloth_xpbd.py")
    set_tests_properties(mlcloth_xpbd PROPERTIES
        ENVIRONMENT "MLCLOTH_XPBD_REFERENCE=$<TARGET_FILE:xpbd_reference>")
else()
    message(STATUS "Mesh converter, capsule and driver tests not registered: no Python 3 interpreter found")
endif()

set(DEFAULT_MODEL_FOR_MESH "${CMAKE_CURRENT_SOURCE_DIR}/../.work/runtime/model_NeuralRes4_NeuralRes4_final.enc")
set(DEFAULT_MESH "${CMAKE_CURRENT_SOURCE_DIR}/../.work/mesh/ch10032_cloth2607.mlmesh")
if(EXISTS "${DEFAULT_MODEL_FOR_MESH}" AND EXISTS "${DEFAULT_MESH}")
    add_test(NAME mlcloth_real_mesh COMMAND mesh_validate "${DEFAULT_MODEL_FOR_MESH}" "${DEFAULT_MESH}")
else()
    message(STATUS "Real mesh validation not registered: run bake_cloth_topology.ps1 first")
endif()

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
