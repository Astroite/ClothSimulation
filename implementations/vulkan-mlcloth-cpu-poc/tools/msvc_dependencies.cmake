# Match the compiler launcher's language regardless of the desktop locale.
# Otherwise Ninja silently misses header dependencies on Chinese Windows hosts.
if(MSVC)
    add_compile_options(/utf-8)
    # This host installs the 2052 compiler resources. Keep this file UTF-8.
    set(CMAKE_CL_SHOWINCLUDES_PREFIX "注意: 包含文件: ")
    set(CMAKE_C_COMPILER_LAUNCHER "${CMAKE_CURRENT_LIST_DIR}/msvc-localized.cmd")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CMAKE_CURRENT_LIST_DIR}/msvc-localized.cmd")
endif()
