

#ifndef DGNN_TEST_DGNN_CLIENT_H
#define DGNN_TEST_DGNN_CLIENT_H




#include <iostream>
#include <grpcpp/grpcpp.h>


#include <vector>

#include "../store/DynamicStore.h"
#include <pthread.h>

#include "dgnn_server.h"
#include "../../cmake/build/dgnn_test.grpc.pb.h"
#include "../../cmake/build/dgnn_test.pb.h"
#include <time.h>
#include <math.h>



using namespace std;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::ClientAsyncResponseReader;
using grpc::CompletionQueue;

using dgnn_test::DgnnProtoService;
using dgnn_test::NodeMessage;
using dgnn_test::EmbGradMessage;
using dgnn_test::NullMessage;



class DGNNClient {
//private:

public:
    std::unique_ptr<DgnnProtoService::Stub> stub_;
    string serverAddress;

    explicit DGNNClient(std::shared_ptr<Channel> channel);

    DGNNClient();

    void init(std::shared_ptr<Channel> channel);

    static void *RunServer(void *address_tmp);

    void startClientServer();


    void init_by_address(std::string address);

    string get_serverAddress();

    void set_serverAddress(string serverAddress);

    void barrier();

    static void *worker_pull_g_dynamic_parallel(void *metaData_void);


    vector<float> sendAccuracy(const vector<int> &v_num, const vector<float> &acc);


    static void *pushVertex2WK(void *metaData);


    static void *getRmtFeatsParallel(void *metaData);






};


struct ReqFeatsMetaData {
    vector<int> *nodes{};
    int workerId{};
    int serverId{};
    DGNNClient *dgnnClient{};
    EmbGradMessage *embGradMessage{};
    string status{};
    int snap_id{};
};

#endif //DGNN_TEST_DGNN_CLIENT_H
