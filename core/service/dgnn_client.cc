



#include "dgnn_client.h"



DGNNClient::DGNNClient(std::shared_ptr<Channel> channel) : stub_(DgnnProtoService::NewStub(channel)) {}

DGNNClient::DGNNClient() = default;

void DGNNClient::init(std::shared_ptr<Channel> channel) {
    stub_ = (DgnnProtoService::NewStub(channel));
}


void *DGNNClient::RunServer(void *address_tmp) {
    string address = *((string *) address_tmp);
    ServiceImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());

    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(2147483647);
    builder.SetMaxSendMessageSize(2147483647);
    builder.SetMaxMessageSize(2147483647);


    std::unique_ptr<Server> server(builder.BuildAndStart());


    std::cout << "Server Listening on" << address << std::endl;
    server->Wait();
}


void DGNNClient::startClientServer() {

    pthread_t serverThread;
    pthread_create(&serverThread, NULL, DGNNClient::RunServer, (void *) &this->serverAddress);
//        ServiceImpl::RunServerByPy(address);

}


void DGNNClient::init_by_address(std::string address) {
    grpc::ChannelArguments channel_args;
    channel_args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, 2147483647);
    std::shared_ptr<Channel> channel = (grpc::CreateCustomChannel(
            address, grpc::InsecureChannelCredentials(), channel_args));
//    DynamicStore::serverip = address;
    stub_ = DgnnProtoService::NewStub(channel);
}


string DGNNClient::get_serverAddress() {
    return this->serverAddress;
}

void DGNNClient::set_serverAddress(string serverAddress) {
    cout << serverAddress << endl;
    this->serverAddress = serverAddress;
}



map<string, float>
DGNNClient::sendAccuracy(float val_acc, float train_acc, float test_acc) {
    ClientContext context;
    AccuracyMessage request;
    AccuracyMessage reply;


    request.set_val_acc(val_acc);
    request.set_train_acc(train_acc);
    request.set_test_acc(test_acc);

    Status status = stub_->sendAccuracy(&context, request, &reply);

    map<string, float> map_acc;
    map_acc.insert(pair<string, float>("val", reply.val_acc_entire()));
    map_acc.insert(pair<string, float>("train", reply.train_acc_entire()));
    map_acc.insert(pair<string, float>("test", reply.test_acc_entire()));

    return map_acc;
}


void DGNNClient::barrier() {
    ClientContext clientContext;
    NullMessage request;
    NullMessage reply;
    Status status = stub_->barrier(&clientContext, request, &reply);
    if (status.ok()) {
//        cout << "okokokokok" << endl;
    } else {
        cout << "server_Barrier false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }
}


void *DGNNClient::worker_pull_g_dynamic_parallel(void *metaData_void) {
    auto metaData = (ReqFeatsMetaData *) metaData_void;
    ClientContext context;
    NullMessage reply;
    EmbGradMessage request = *metaData->embGradMessage;
    auto *dgnnClient = metaData->dgnnClient;

    Status status = dgnnClient->stub_->setAndSendGDynamic(&context, request, &reply);
    if (status.ok()) {
//        unique_lock<mutex> lck(ThreadUtil::mtx_setAndSendG_for_count);
//        ThreadUtil::count_setAndSendG++;
//        lck.unlock();
    } else {
        cout << "worker_pull_g_parallel false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
        exit(-1);
    }

}


void *DGNNClient::pushVertex2WK(void *metaData_void) {
    auto metaData = (ReqFeatsMetaData *) metaData_void;
    auto nodes = *metaData->nodes;
    DGNNClient *dgnnClient = metaData->dgnnClient;
    int workerID = metaData->workerId;
    ClientContext context;
    NodeMessage request;
    NullMessage reply;
    request.set_wid(workerID);
    request.mutable_nodes()->Add(nodes.begin(), nodes.end());
    Status status = dgnnClient->stub_->pushVertex2WK(&context, request, &reply);
    if (status.ok()) {
        delete metaData;
    } else {
        cout << "error pushVertex2WK" << endl;
    }
}


void *DGNNClient::getRmtFeatsParallel(void *metaData_void) {

    auto metaData = (ReqFeatsMetaData *) metaData_void;
    int workerId = metaData->workerId;
    DGNNClient *dgnnClient = metaData->dgnnClient;
    auto request=metaData->embGradMessage;
    request->set_workerid(workerId);


    //    cout<<"node size:"<<nodes.size()<<",buf1 size:"<<nodes_buf.size<<endl;

    // 去服务器中获取嵌入,这里建立的是每个worker的channel

    ClientContext context;

    EmbGradMessage reply;
    // 构建request



    Status status = dgnnClient->stub_->workerPullEmbDynamic(&context, *request, &reply);
    if (status.ok()) {


    } else {
        cout << "pull needed remote features false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
        exit(-1);
    }


}




