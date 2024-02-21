# DynaHB: Communication-Avoiding Asynchronous Distributed Framework with Hybrid Batches for Dynamic GNN Training



Note that, /mnt/data/cora is the common-shared nfs directory. 
Also, you can create the directory in the local to simulate the 
distributed environment. 

##**_How to Install ADGNN_**

If you just want to using ADGNN to build your own GNN models, you need to install the python dependencies:
`python3.6` and `requirements` in "python/requirements.txt" on Ubuntu16.04 (other versions of Ubuntu can also work).
Then use "ldd cmake/build/example2.cpython-36m-x86_64-linux-gnu.so" to detect if all dependencies are satisfied. 

**_otherwise_**:

If you intend to modify the core codes of ADGNN in c++, beyond the python dependencies, you need to install `cmake, grpc, protobuf, pybind11`

```
mkdir cmake/build && cd cmake/build
cmake ../..
make
```

If build successfully, it will generate new 4 grpc and protobuf files (".grpc.pb.cc and .h","pb.cc and .h" )
 and dynamic link libraries lib/pb11_ec.cpython-36m-x86_64-linux-gnu.so and lib/datatrans.cpython-36m-x86_64-linux-gnu.so. Then, you can run EC-Graph
 following the instructions in "How to Run an Example"


You can use docker to run ADGNN on the servers, please see details in Dockerfile.

##**_How to Run an Example_**

1: Install a distributed file system, e.g., NFS, HDFS. Set the shared-directory as "/mnt/data". 
If you just want to run ADGNN on a single-machine, you can just "mkdir /mnt/data" without installing NFS.

2: Processing the data format to the ADGNN format by using programs in "python/data_processing".
```
Two files will be created (all separators are "\t"). 

featsClass.txt (id   feat (dim = 5)   class):
0 1 0 1 1 1 0
1 0 1 1 0 1 1
2 0 0 0 1 1 0

edges.txt (src   dst)
0   1
1   2
0   2
``` 

3: Move these two files to their directory "/mnt/data/cora"

4: Set the number of workers and servers in "python/context/context.py"

5: Run 1 worker and 1 server as an example
```
Run "python/example/distgnn/dist_start.py" with "--role_id=server,0"
Run "python/example/distgnn/dist_start.py" with "--role_id=worker,0"
```
The other settings are shown at the beginning "argument list".



