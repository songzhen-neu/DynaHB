

#include "DynamicGraphBuild.h"
#include "../store/DynamicStore.h"
//#include "../store/ServerStore.h"


void buildRmtVertexNum(const string &mode){
    auto worker_id=DynamicStore::worker_id;
    auto snap_num=DynamicStore::snap_num[mode];
    DynamicStore::rmt_vertex_num.insert(make_pair(mode,vector<int>(snap_num)));
    for(int i=0;i<DynamicStore::rmt_neis[mode].size();i++){
        for(auto& elem: DynamicStore::rmt_neis[mode][i]){
            if(elem.first!=worker_id){
                DynamicStore::rmt_vertex_num[mode][i]+=elem.second.size();
            }
        }
    }
}

void DynamicGraphBuild::transDataToCpp(int worker_id,int worker_num,int feat_size,const unordered_map<int,int> v2wk,const vector<map<int,int>> &train_old2new_maps,const vector<map<int,set<int>>> &train_rmt_neis,
                                       const vector<map<int,int>> &test_old2new_maps,const vector<map<int,set<int>>> &test_rmt_neis ) {


    DynamicStore::worker_id = worker_id;
    DynamicStore::worker_num = worker_num;
//    ServerStore::worker_num = worker_num;
//    DynamicStore::feat_size = feat_size;
    for(auto& elem:v2wk){
        DynamicStore::v2wk.insert(make_pair(elem.first,elem.second));
    }

    DynamicStore::old2new_maps.insert(make_pair("train", train_old2new_maps));
    DynamicStore::old2new_maps.insert(make_pair("test", test_old2new_maps));
    DynamicStore::rmt_neis.insert(make_pair("train", train_rmt_neis));
    DynamicStore::rmt_neis.insert(make_pair("test", test_rmt_neis));

    DynamicStore::snap_num.insert(make_pair("train",train_old2new_maps.size()));
    DynamicStore::snap_num.insert(make_pair("test",test_old2new_maps.size()));

    // calculate the number of remote vertices for allocating space for results returned
    int rmtvnum_4train=0;
    buildRmtVertexNum("train");
    buildRmtVertexNum("test");





}

void DynamicGraphBuild::transBasicDataToCpp(int worker_id,int worker_num) {

    DynamicStore::worker_id = worker_id;
    DynamicStore::worker_num = worker_num;
//    DynamicStore::local_vertexs = local_nodes;
}


void DynamicGraphBuild::pushLocalFeats(py::array_t<float> &local_feats){
    DynamicStore::local_feats = local_feats;
}



void DynamicGraphBuild::printmessage(){


//    cout<<"worker_id:"<< DynamicStore::worker_id<<endl;
//    cout<<"DynamicStore::worker_num"<<DynamicStore::worker_num<<endl;
//    cout<<"train_old2new_maps:"<<endl;
//    for (const auto& innerMap : DynamicStore::old2new_maps["train"]) {
//        for (const auto& pair : innerMap) {
//            int key = pair.first;
//            int value = pair.second;
//            std::cout << "Key: " << key << ", Value: " << value <<'\t';
//        }
//        cout<<endl;
//    }
//
//    cout << "train_rmt_neis:" << endl;
//    for (const auto& innerMap : DynamicStore::rmt_neis["train"]) {
//        for (const auto& entry : innerMap) {
//            int key = entry.first;
//            const set<int>& value = entry.second;
//            cout << "Key: " << key << ", Values: ";
//            for (const int& element : value) {
//                std::cout << element << " ";
//            }
//            std::cout << std::endl;
//        }
//    }

}


