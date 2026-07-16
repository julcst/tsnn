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

    # Precompile each module to a `.slang-module` binary cache next to its source in the
    # deploy dir. Slang's session automatically prefers an up-to-date `.slang-module` over
    # recompiling the matching `.slang` from scratch, which skips re-parsing/re-checking
    # TSNN's heavily-generic (CoopVec/HashGrid/RQS) source on every Mogwai launch. Each
    # module is compiled with `-no-codegen` (front-end/IR only, matching how Falcor's own
    # ProgramManager defers codegen) so the cache is target-agnostic (D3D12/Vulkan share it).
    #
    # Done as a single serialized step (precompile_modules.cmake) rather than one
    # add_custom_command per module: slangc resolves a consuming module's imports (e.g.
    # MLP.slang -> CoopVec.slang-module) against whatever is on disk at invocation time, so
    # running these as independent parallel ninja jobs lets one module read another's
    # `.slang-module` mid-write and fail to resolve generic witnesses out of the torn file.
    # This step is best-effort (see precompile_modules.cmake): a module that fails to
    # precompile is skipped with a warning instead of failing the build, and Slang just
    # compiles it from source at that point like it would without this cache at all.
    set(SLANGC_EXECUTABLE "${SLANG_DIR}/bin/slangc")
    set(TSNN_MODULE_STAMP "")
    if(EXISTS ${SLANGC_EXECUTABLE})
        set(TSNN_MODULE_STAMP "${CMAKE_CURRENT_BINARY_DIR}/tsnn_modules.stamp")
        add_custom_command(
            OUTPUT ${TSNN_MODULE_STAMP}
            COMMAND ${CMAKE_COMMAND} -D SLANGC_EXECUTABLE=${SLANGC_EXECUTABLE} -D DST_DIR=${TSNN_SHADER_DST_DIR}
                    -D STAMP_FILE=${TSNN_MODULE_STAMP} -P "${CMAKE_CURRENT_SOURCE_DIR}/precompile_modules.cmake"
            DEPENDS ${TSNN_SHADER_OUTPUTS} "${CMAKE_CURRENT_SOURCE_DIR}/precompile_modules.cmake"
            COMMENT "TSNN: precompiling modules"
        )
    else()
        message(WARNING "TSNN: slangc not found at ${SLANGC_EXECUTABLE}, skipping module precompilation.")
    endif()

    add_custom_target(tsnn_shaders ALL DEPENDS ${TSNN_SHADER_OUTPUTS} ${TSNN_MODULE_STAMP})
endif()

# Interface target so dependents can find the slang source tree.
add_library(tsnn INTERFACE)
target_include_directories(tsnn INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
