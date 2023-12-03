

#ifndef TGNN_DYNAMICGRAPHBUILD_H
#define TGNN_DYNAMICGRAPHBUILD_H



#include <vector>
#include <map>
#include <set>
#include <pybind11/pybind11.h>
#include "pybind11/numpy.h"

using namespace std;
namespace py = pybind11;

class DynamicGraphBuild {
public:

    void transDataToCpp(int worker_id,int worker_num,int feat_size,const unordered_map<int,int> v2wk, const vector<map<int,int>> &train_old2new_maps,const vector<map<int,set<int>>> &train_rmt_neis,
                        const vector<map<int,int>> &test_old2new_maps,const vector<map<int,set<int>>> &test_rmt_neis );
    void pushLocalFeats(py::array_t<float> &local_feats);
    void printmessage();
    void transBasicDataToCpp(int worker_id,int worker_num);
//    map<int,int> getEncode();

};


#endif //TGNN_DYNAMICGRAPHBUILD_H
