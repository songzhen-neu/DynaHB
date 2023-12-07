

#include "router.h"
#include <cstring>

Router::Router() {}

vector<DGNNClient *> Router::dgnnWorkerRouter;




void Router::initWorkerRouter(map<int, string> &dgnnWorkerAddress) {

    for (auto &address:dgnnWorkerAddress) {
        DGNNClient *dgnnClient = new DGNNClient();
        dgnnClient->init_by_address(address.second);
        dgnnWorkerRouter.push_back(dgnnClient);
        cout << address.second << endl;
    }
//    cout<<dgnnWorkerRouter[1]->add1()<<endl;
}


vector<vector<float>> initGradTmp(int node_num, int dim_num) {
    vector<vector<float>> grad_tmp(node_num);
    for (int i = 0; i < node_num; i++) {
        vector<float> vec_tmp(dim_num);
        grad_tmp[i] = vec_tmp;
    }
    return grad_tmp;
}



py::array_t<float>
Router::setAndSendGDynamic(int snap_id, const string &status, const map<int, vector<int>> &vertex_push,
                           map<int, int> &rcv_nei_num, const py::array_t<float> &emb_grads) {
    py::buffer_info buf = emb_grads.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("ids dim size!=1 or embs dim size!=2");
    }

    int node_num = buf.shape[0];
    auto *ptr = (float *) buf.ptr;
    int feat_size = buf.shape[1];
    auto &rmt_nei = DynamicStore::rmt_neis[status][snap_id];
    int local_node_num = DynamicStore::local_feats.request().shape[0];

    DynamicStore::vertex_push = vertex_push;

    auto grad_tmp = initGradTmp(local_node_num, feat_size);
    vector<pthread_t> pthreads(DynamicStore::worker_num);
    auto &o2n_map = DynamicStore::old2new_maps[status][snap_id];
//    auto &n2o_map = DynamicStore::new2old_maps[status][snap_id];
    // {worker:{snap_id:vec}}
    map<int, map<int, vector<float>>> message;
    vector<EmbGradMessage> embgrad_message_vec(DynamicStore::worker_num);



    for (int i = 0; i < DynamicStore::worker_num; i++) {
        embgrad_message_vec[i].set_featsize(feat_size);
        embgrad_message_vec[i].set_snapid(snap_id);
//        embgrad_message_vec[i].set_graph_mode(status);
        embgrad_message_vec[i].set_workerid(DynamicStore::worker_id);
        embgrad_message_vec[i].set_status(status);

    }



    for (int i = 0; i < local_node_num; i++) {
        float *grad_tmp_ptr = &grad_tmp[i][0];
        copy(ptr + i * feat_size, ptr + (i + 1) * feat_size, grad_tmp_ptr);
    }


    for (int i = 0; i < DynamicStore::worker_num; i++) {
        if (i != DynamicStore::worker_id) {
            int start_index = local_node_num;

            for (int j = start_index; j < start_index + rcv_nei_num[i]; j++) {
                embgrad_message_vec[i].mutable_embs()->Add(ptr + j * feat_size, ptr + (j + 1) * feat_size);
            }
            start_index = start_index + rcv_nei_num[i];

        }
    }





    DynamicStore::local_emb_grad_agg[snap_id] = grad_tmp;

    dgnnWorkerRouter[0]->barrier();


    for (int i = 0; i < DynamicStore::worker_num; i++) {
        if (i != DynamicStore::worker_id) {
//            pthread_t p;
            auto *metaData = new ReqFeatsMetaData;
            metaData->embGradMessage = &embgrad_message_vec[i];
            metaData->dgnnClient = dgnnWorkerRouter[i];
            pthread_create(&pthreads[i], NULL, DGNNClient::worker_pull_g_dynamic_parallel, (void *) metaData);
        }
    }

    pthread_vec_join(pthreads, DynamicStore::worker_id, DynamicStore::worker_num);


    dgnnWorkerRouter[0]->barrier();

    // return the aggregate of local_emb_grad
    // the order follows the old2new_map of layer_id
    auto result = py::array_t<float>(local_node_num * feat_size);
    result.resize({local_node_num, feat_size});
    py::buffer_info buf_result = result.request();
    float *ptr_result = (float *) buf_result.ptr;
    for (int i = 0; i < local_node_num; i++) {
        copy(DynamicStore::local_emb_grad_agg[snap_id][i].begin(), DynamicStore::local_emb_grad_agg[snap_id][i].end(),
             ptr_result + i * feat_size);
//        for (int j = 0; j < dim_num; j++) {
//            ptr_result[i * dim_num + j] = WorkerStore::local_emb_grad_agg[layer_id][i][j];
//        }
    }

//    dgnnServerRouter[0]->server_Barrier();
    return result;

}


pair<vector<int>, map<int, vector<int>>> Router::pushVertex(map<int, vector<int>> &vertex2wk) {


    vector<pthread_t> pthreads(DynamicStore::worker_num);
    for (int i = 0; i < DynamicStore::worker_num; i++) {
        if (DynamicStore::worker_id != i) {
            auto *metaData = new ReqFeatsMetaData;
            metaData->dgnnClient = dgnnWorkerRouter[i];
            metaData->nodes = &vertex2wk[i];
            metaData->workerId = DynamicStore::worker_id;
            pthread_create(&pthreads[i], NULL, DGNNClient::pushVertex2WK, (void *) metaData);
        }
    }
    pthread_vec_join(pthreads, DynamicStore::worker_id, DynamicStore::worker_num);
    dgnnWorkerRouter[0]->barrier();
    vector<int> rmt_neis;
    for (int i = 0; i < DynamicStore::worker_num; i++) {
        if (i != DynamicStore::worker_id) {
            auto &vertices = DynamicStore::rmt_vertex_need_encode[i];
            rmt_neis.insert(rmt_neis.end(), vertices.begin(), vertices.end());
        }

    }

//    auto result_tmp=map<int,vector<int>>(DynamicStore::rmt_vertex_need_encode);
//    dgnnWorkerRouter[0]->barrier();
//    return make_pair(rmt_neis, aa);

    return make_pair(rmt_neis, DynamicStore::rmt_vertex_need_encode);


}





py::array_t<float> Router::getRmtFeats(const string &status, int snap_id, map<int, vector<int>> &vertex_push) {
    Router::dgnnWorkerRouter[0]->barrier();

//    auto local_feat_ptr =  (float *) DynamicStore::local_feats.request().ptr;

    py::buffer_info buf = DynamicStore::local_feats.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("ids dim size!=1 or embs dim size!=2");
    }
    auto *ptr = (float *) buf.ptr;
    int feat_size = buf.shape[1];
    auto &rmt_nei = DynamicStore::rmt_neis[status][snap_id];
    int local_node_num = DynamicStore::local_feats.request().shape[0];

    vector<EmbGradMessage> embgrad_message_vec(DynamicStore::worker_num);
    for (int i = 0; i < DynamicStore::worker_num; i++) {
        if (i != DynamicStore::worker_id) {
            for (auto push_id:vertex_push[i]) {
                embgrad_message_vec[i].mutable_embs()->Add(ptr + push_id * feat_size, ptr + (push_id + 1) * feat_size);
            }
        }
    }

    vector<pthread_t> pthreads(DynamicStore::worker_num);
    for (int i = 0; i < DynamicStore::worker_num; i++) {
        if (i != DynamicStore::worker_id) {
            pthread_t p;
            auto *metaData = new ReqFeatsMetaData;
            metaData->embGradMessage = &embgrad_message_vec[i];
            metaData->dgnnClient = dgnnWorkerRouter[i];
            metaData->workerId = DynamicStore::worker_id;
            pthread_create(&pthreads[i], NULL, DGNNClient::getRmtFeatsParallel, (void *) metaData);
        }
    }


    pthread_vec_join(pthreads, DynamicStore::worker_id, DynamicStore::worker_num);
    dgnnWorkerRouter[0]->barrier();

    int total_size = 0;
    for (auto &elem:DynamicStore::message_fp) {
        total_size += elem.second.size();
    }
//    cout << feat_size << "," << total_size << endl;
    auto result = py::array_t<float>({total_size / feat_size, feat_size});
    auto ptr_result = (float *) result.request().ptr;
//    auto ptr_cur = ptr_result;

//    for (auto elem:DynamicStore::message_fp) {
//        for (auto i:elem.second) {
//            cout << i << ",";
//        }
//        cout << endl;
//    }
    vector<float> tmp;

    for (int i = 0; i < DynamicStore::worker_num; i++) {
        if (i != DynamicStore::worker_id) {
            tmp.insert(tmp.end(),DynamicStore::message_fp[i].begin(),DynamicStore::message_fp[i].end());
        }
    }
//    for(int i=0;i<tmp.size();i++){
//        cout<<tmp[i]<<endl;
//    }
    copy(tmp.begin(),tmp.end(),ptr_result);
    return result;
}



void Router::pthread_vec_join(const vector<pthread_t> &pthreads,int worker_id,int worker_num) {
    for (int i = 0; i <worker_num; i++) {
        if (worker_id != i) {
            pthread_join(pthreads[i], NULL);
        }
    }
}


//vector<int,int> Router::updateRLStrategy(const vector<int> &window_batch_size){
//
//}
