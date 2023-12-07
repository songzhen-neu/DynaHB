

#ifndef DGNN_TEST_ROUTER_H
#define DGNN_TEST_ROUTER_H
//#include <vector>
#include "dgnn_client.h"
#include <map>
#include <string>
#include <vector>
#include <time.h>


using namespace std;

class Router {
public:
    static vector<DGNNClient*> dgnnWorkerRouter;


    Router();
    void initWorkerRouter(map<int,string> &dgnnWorkerAddress);
    py::array_t<float> setAndSendGDynamic(int snap_id, const string &status,const map<int,vector<int>> &vertex_push, map<int,int> &rcv_nei_num, const py::array_t<float> &emb_grads);
    pair<vector<int>,map<int,vector<int>>> pushVertex(map<int,vector<int>> &vertex2wk);

    static void pthread_vec_join(const vector<pthread_t> &pthreads,int worker_id,int worker_num);

    py::array_t<float> getRmtFeats(const string &status,int snap_id, map<int,vector<int>>& vertex_push);
    vector<int,int> updateRLStrategy(const vector<int> &window_batch_size);

};


#endif //DGNN_TEST_ROUTER_H
