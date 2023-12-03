#aux_source_directory(. DIR_LIB_SRCS)
#add_library(resource ${DIR_LIB_SRCS})

pybind11_add_module(pb11_ec pb11_ec.cc)
target_link_libraries(pb11_ec PRIVATE store)
target_link_libraries(pb11_ec PRIVATE service)
target_link_libraries(pb11_ec PRIVATE graphbuild)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)