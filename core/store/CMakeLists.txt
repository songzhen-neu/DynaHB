aux_source_directory(. DIR_LIB_SRCS)
add_library(store ${DIR_LIB_SRCS})
target_link_libraries(store util service)