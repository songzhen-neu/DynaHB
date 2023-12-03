

#ifndef DGNN_TEST_THREADUTIL_H
#define DGNN_TEST_THREADUTIL_H
#include<pthread.h>
#include <condition_variable>
#include <unistd.h>
#include <mutex>
#include <vector>
#include <atomic>
using namespace std;

class ThreadUtil {
public:


    static mutex mtx_barrier;
    static condition_variable cv_barrier;
    static int count_worker_for_barrier;


    static mutex mtx_barrier_server;
    static condition_variable cv_barrier_server;
    static int count_worker_for_barrier_server;



    static mutex mtx_common;






};


#endif //DGNN_TEST_THREADUTIL_H
