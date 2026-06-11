cmake_minimum_required(VERSION 3.21)

# TSNN - Typed Slang Neural Networks
# Framework-agnostic Slang library for differentiable neural networks.
# Mirrors the role of tiny-cuda-nn but targets the Slang/Vulkan/HLSL ecosystem.
set(TSNN_SHADER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/TSNN")

# Install shaders into the Falcor shader output directory when used inside Falcor.
# When FALCOR_SHADER_OUTPUT_DIRECTORY is not set (standalone usage), skip the copy target.
if(DEFINED FALCOR_SHADER_OUTPUT_DIRECTORY)
    set(TSNN_SHADER_DST_DIR "${FALCOR_SHADER_OUTPUT_DIRECTORY}/TSNN")

    # Recurse so the package's subdirectories (Utils, Optimizers, Encodings,
    # Flows, ...) are mirrored.  The dotted import paths (e.g.
    # `import TSNN.Encodings.HashGrid3D`) require the on-disk tree to be
    # preserved, so we copy each file to its matching relative location.
    file(GLOB_RECURSE TSNN_SHADERS
        CONFIGURE_DEPENDS
        "${TSNN_SHADER_SRC_DIR}/*.slang"
        "${TSNN_SHADER_SRC_DIR}/*.slangh"
    )

    set(TSNN_SHADER_OUTPUTS "")

    foreach(SRC ${TSNN_SHADERS})
        file(RELATIVE_PATH REL "${TSNN_SHADER_SRC_DIR}" "${SRC}")
        set(DST "${TSNN_SHADER_DST_DIR}/${REL}")
        add_custom_command(
            OUTPUT ${DST}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC} ${DST}
            DEPENDS ${SRC}
            COMMENT "TSNN: copying ${REL}"
        )
        list(APPEND TSNN_SHADER_OUTPUTS ${DST})
    endforeach()

    add_custom_target(tsnn_shaders ALL DEPENDS ${TSNN_SHADER_OUTPUTS})
endif()

# Interface target so dependents can find the slang source tree.
add_library(tsnn INTERFACE)
target_include_directories(tsnn INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
