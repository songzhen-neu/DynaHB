
#include "DynamicStore.h"

int DynamicStore::worker_id;
int DynamicStore::worker_num;

map<string,int> DynamicStore::snap_num;
map<string, vector<map<int,int>>> DynamicStore::old2new_maps;

map<string, vector<map<int,set<int>>>> DynamicStore::rmt_neis;
map<string, vector<int>> DynamicStore::rmt_vertex_num;

py::array_t<float> DynamicStore::local_feats;

map<int,vector<vector<float>>> DynamicStore::local_emb_grad_agg;

unordered_map<int,int> DynamicStore::v2wk;

map<int,vector<int>> DynamicStore::rmt_vertex_need_encode;

map<int,vector<int>> DynamicStore::vertex_push;


map<int,vector<float>> DynamicStore::message_fp;


vector<float> DynamicStore::accuracy_total;
vector<int> DynamicStore::acc_vertex_num ;

//vector<int> DynamicStore::window_batch_size;