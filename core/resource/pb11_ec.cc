

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../service/dgnn_server.h"
#include "../service/dgnn_client.h"
#include "pybind11/numpy.h"
#include "../service/router.h"
#include "../graph_build/DynamicGraphBuild.h"



using namespace std;
namespace py = pybind11;




PYBIND11_MODULE(pb11_ec, m) {

    m.doc() = "pybind11 example plugin";



    py::class_<Router>(m,"Router")
            .def(py::init<>())
            .def("initWorkerRouter",&Router::initWorkerRouter)
            .def("getRmtFeats",&Router::getRmtFeats)
            .def("setAndSendGDynamic",&Router::setAndSendGDynamic)
            .def("pushVertex",&Router::pushVertex);
//            .def("updateRLStrategy",&Router::updateRLStrategy);


    py::class_<DynamicGraphBuild>(m,"DynamicGraphBuild")
            .def(py::init<>())
            .def("transDataToCpp",&DynamicGraphBuild::transDataToCpp)
            .def("pushLocalFeats",&DynamicGraphBuild::pushLocalFeats)
            .def("transBasicDataToCpp",&DynamicGraphBuild::transBasicDataToCpp)
            .def("printmessage",&DynamicGraphBuild::printmessage);


    // 创建client
    py::class_<DGNNClient>(m,"DGNNClient")
            .def(py::init<>())
            .def("init_by_address",&DGNNClient::init_by_address)
            .def("startClientServer",&DGNNClient::startClientServer)
            .def("barrier",&DGNNClient::barrier)
            .def("sendAccuracy",&DGNNClient::sendAccuracy)
            .def_property("serverAddress",&DGNNClient::get_serverAddress,&DGNNClient::set_serverAddress);

}



