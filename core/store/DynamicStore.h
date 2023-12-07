


#ifndef TGNN_DYNAMICSTORE_H

#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include "pybind11/numpy.h"

using namespace std;
namespace py = pybind11;

class DynamicStore{
public:



    static int worker_id;
    static int worker_num;

    static map<string,int> snap_num;
    // mode->snap->worker->rmt_v
    static map<string, vector<int>> rmt_vertex_num;
    static map<string, vector<map<int,int>>> old2new_maps;
    // mode->snap->worker->rmt_v
    static map<string, vector<map<int,set<int>>>> rmt_neis;
    static py::array_t<float> local_feats;

    //snap_id->grad
    static map<int,vector<vector<float>>> local_emb_grad_agg;
    static unordered_map<int,int> v2wk;

    //worker->vertex
    static map<int,vector<int>> rmt_vertex_need_encode;
    static map<int,vector<int>> vertex_push;


    static map<int,vector<float>> message_fp;


    static vector<float> accuracy_total ;
    static vector<int> acc_vertex_num ;

//    static vector<int> window_batch_size;



};
#define TGNN_DYNAMICSTORE_H



#endif //TGNN_DYNAMICSTORE_H


