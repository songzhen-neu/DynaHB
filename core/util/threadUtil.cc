

#include "threadUtil.h"



mutex ThreadUtil::mtx_barrier;
condition_variable ThreadUtil::cv_barrier;
int ThreadUtil::count_worker_for_barrier;

mutex ThreadUtil::mtx_barrier_server;
condition_variable ThreadUtil::cv_barrier_server;
int ThreadUtil::count_worker_for_barrier_server=0;



mutex ThreadUtil::mtx_common;






