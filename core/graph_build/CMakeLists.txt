aux_source_directory(. DIR_LIB_SRCS)
add_library(graphbuild ${DIR_LIB_SRCS})
target_link_libraries(graphbuild service)