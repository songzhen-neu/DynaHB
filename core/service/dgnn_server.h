

#ifndef DGNN_TEST_DGNN_SERVER_H
#define DGNN_TEST_DGNN_SERVER_H

#include <iostream>
#include <grpcpp/grpcpp.h>
#include<grpcpp/health_check_service_interface.h>
#include<grpcpp/ext/proto_server_reflection_plugin.h>
//#include "../store/WorkerStore.h"
#include "../../cmake/build/dgnn_test.grpc.pb.h"
#include "../../cmake/build/dgnn_test.pb.h"
//#include "../partition/GeneralPartition.h"
#include "../util/threadUtil.h"
//#include "../store/ServerStore.h"
#include "../store/DynamicStore.h"
#include <cmath>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <google/protobuf/repeated_field.h>
using grpc::Server;
using grpc::ServerContext;
using grpc::Status;
using grpc::ServerBuilder;


// here is proto defining package name. and the classes belong to this package
using dgnn_test::DgnnProtoService;
using dgnn_test::EmbGradMessage;
using dgnn_test::AccuracyMessage;
using dgnn_test::NodeMessage;
using dgnn_test::NullMessage;

class ServiceImpl final:public DgnnProtoService::Service{
public:


    Status barrier(
            ServerContext* context,const NullMessage* request,
            NullMessage* reply) override;

    Status workerPullEmbDynamic(
            ServerContext* context,const EmbGradMessage* request,
            EmbGradMessage* reply) override;


    Status sendAccuracy(ServerContext *context,const AccuracyMessage *request,
                        AccuracyMessage *reply) override;


//    static void RunServerByPy(const string& address);
    static void barrier_server(int barrier_num);

    Status setAndSendGDynamic(ServerContext *context, const EmbGradMessage *request, NullMessage *reply) override;

    Status pushVertex2WK(ServerContext *context,const NodeMessage *request,NullMessage *reply) override;


};




#endif //DGNN_TEST_DGNN_SERVER_H







