# Precompiles every TSNN `.slang` module (already deployed under DST_DIR) into a
# `.slang-module` binary cache next to it. Run sequentially in a single build step
# (see external/tsnn/CMakeLists.txt) so a module's imports are never read from a
# concurrently-written, torn `.slang-module` file produced by a parallel ninja job.
#
# Best-effort: a module that fails to precompile is skipped with a warning rather than
# failing the build. Slang falls back to compiling straight from source for any module
# without a valid `.slang-module` next to it, so a skipped module simply doesn't benefit
# from the cache instead of blocking the whole project.
#
# Expected -D args: SLANGC_EXECUTABLE, DST_DIR, STAMP_FILE

file(GLOB_RECURSE MODULE_SOURCES "${DST_DIR}/*.slang")

foreach(SRC ${MODULE_SOURCES})
    execute_process(
        COMMAND ${SLANGC_EXECUTABLE} -I ${DST_DIR}/.. -no-codegen -o ${SRC}-module ${SRC}
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE OUTPUT
        ERROR_VARIABLE OUTPUT
    )
    if(NOT RESULT EQUAL 0)
        message(WARNING "TSNN: failed to precompile ${SRC}, will compile from source instead:\n${OUTPUT}")
        file(REMOVE "${SRC}-module")
    endif()
endforeach()

file(TOUCH "${STAMP_FILE}")
