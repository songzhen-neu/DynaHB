
#include <sys/time.h>
#include "dgnn_server.h"
// define the final ServiceImpl class (it cannot be inherited); and it extends the class Service by 'public' manner,
// it can access any public member in the parent class, expect private members.



Status ServiceImpl::workerPullEmbDynamic(
        ServerContext *context, const EmbGradMessage *request, EmbGradMessage *reply) {

//    clock_t start = clock();
//    auto mode = request->status(); // mom mv none

    auto from_wid = request->workerid();
    auto &emb = request->embs();

    unique_lock<mutex> mutex(ThreadUtil::mtx_common);
    vector<float> vec_tmp(emb.size());
    copy(emb.begin(), emb.end(), vec_tmp.begin());
    DynamicStore::message_fp[from_wid] = vec_tmp;

    mutex.unlock();


    return Status::OK;
}


Status ServiceImpl::barrier(
        ServerContext *context, const NullMessage *request, NullMessage *reply) {
    unique_lock<mutex> lck(ThreadUtil::mtx_barrier);
    ThreadUtil::count_worker_for_barrier++;
    if (ThreadUtil::count_worker_for_barrier == DynamicStore::worker_num) {
        ThreadUtil::count_worker_for_barrier = 0;
        ThreadUtil::cv_barrier.notify_all();

    } else {
        ThreadUtil::cv_barrier.wait(lck);
    }
    return Status::OK;
}


//void ServiceImpl::RunServerByPy(const string &address) {
//    ServiceImpl service;
//
////    DynamicStore::serverip = serverId;
//    grpc::EnableDefaultHealthCheckService(true);
//    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
//    ServerBuilder builder;
//    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
//
//    builder.RegisterService(&service);
//    builder.SetMaxReceiveMessageSize(2147483647);
//    builder.SetMaxSendMessageSize(2147483647);
//    builder.SetMaxMessageSize(2147483647);
//
//
//    std::unique_ptr<Server> server(builder.BuildAndStart());
//
//
//    std::cout << "Server Listening on " << address << std::endl;
//    std::cout << "if signal 11, please check ip :" << address << std::endl;
//    server->Wait();
//
//}


Status ServiceImpl::sendAccuracy(ServerContext *context, const AccuracyMessage *request,
                                 AccuracyMessage *reply) {

    if(request->worker_id()==0){
        DynamicStore::accuracy_total.clear();
        DynamicStore::acc_vertex_num.clear();
        for(int i=0;i<request->v_num_size();i++){
            DynamicStore::accuracy_total.push_back(0);
            DynamicStore::acc_vertex_num.push_back(0);
        }
    }

    barrier_server(DynamicStore::worker_num);

    unique_lock<mutex> lck(ThreadUtil::mtx_common);
    for(int i=0;i<request->v_num_size();i++){
        DynamicStore::accuracy_total[i]+=request->acc(i)*request->v_num(i);
        DynamicStore::acc_vertex_num[i]+=request->v_num(i);
    }
    lck.unlock();

    barrier_server(DynamicStore::worker_num);

    if(request->worker_id()==0) {
        for(int i=0;i<request->v_num_size();i++){
            DynamicStore::accuracy_total[i]/=DynamicStore::acc_vertex_num[i];

        }
    }

    barrier_server(DynamicStore::worker_num);

    reply->mutable_acc_total()->Add(DynamicStore::accuracy_total.begin(),DynamicStore::accuracy_total.end());

    return Status::OK;

}


void ServiceImpl::barrier_server(int barrier_num) {
    unique_lock<mutex> lck_barrier(ThreadUtil::mtx_barrier_server);
    ThreadUtil::count_worker_for_barrier_server++;
    if (ThreadUtil::count_worker_for_barrier_server == barrier_num) {
        ThreadUtil::count_worker_for_barrier_server = 0;
        ThreadUtil::cv_barrier_server.notify_all();
    } else {
        ThreadUtil::cv_barrier_server.wait(lck_barrier);
    }
}


Status ServiceImpl::setAndSendGDynamic(ServerContext *context, const EmbGradMessage *request, NullMessage *reply) {
    int dim_num = request->featsize();
    const string &status = request->status();
    int snap_id = request->snapid();
    int from_wid = request->workerid();

    auto &o2n_map = DynamicStore::old2new_maps[status][snap_id];

    unique_lock<mutex> lck(ThreadUtil::mtx_common);
    auto &loc_embgrad_agg_lay = DynamicStore::local_emb_grad_agg[snap_id];
    auto &vertex_push = DynamicStore::vertex_push;

    for (int i = 0; i < vertex_push[from_wid].size(); i++) {
        int new_id = vertex_push[from_wid][i];
        auto &loc_embgrad_agg_node = loc_embgrad_agg_lay[new_id];
        for (int j = 0; j < dim_num; j++) {
            loc_embgrad_agg_node[j] += request->embs(i * dim_num + j);
        }
    }


    lck.unlock();
    return Status::OK;
}


Status ServiceImpl::pushVertex2WK(ServerContext *context, const NodeMessage *request, NullMessage *reply) {


    barrier_server(DynamicStore::worker_num - 1);

    unique_lock<mutex> lck_merge(ThreadUtil::mtx_common);
    if (!DynamicStore::rmt_vertex_need_encode[request->wid()].empty()) {
        DynamicStore::rmt_vertex_need_encode[request->wid()].clear();
    }
    DynamicStore::rmt_vertex_need_encode[request->wid()].insert(
            DynamicStore::rmt_vertex_need_encode[request->wid()].begin(), request->nodes().begin(),
            request->nodes().end());

    lck_merge.unlock();


    barrier_server(DynamicStore::worker_num - 1);

    return Status::OK;

}
