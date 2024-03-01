# DynaHB: Communication-Avoiding Asynchronous Distributed Framework with Hybrid Batches for Dynamic GNN Training


##**_How to Install DynaHB_**

If you just want to using DynaHB to build your own DGNN models, you need to install the python dependencies:
`python3.9` and `requirements` in "python/requirements.txt" on Ubuntu18.04 (other versions of Ubuntu can also work).
Then use "ldd cmake/build/pb11_ec.cpython-39-x86_64-linux-gnu.so" to detect if all dependencies are satisfied. 

**_otherwise_**:

If you intend to modify the core codes of DynaHB in c++, beyond the python dependencies, you need to install `cmake, grpc, protobuf, pybind11`

```
mkdir cmake/build && cd cmake/build
cmake ../..
make
```

If build successfully, it will generate new 4 grpc and protobuf files (".grpc.pb.cc and .h","pb.cc and .h" )
 and dynamic link libraries lib/pb11_ec.cpython-39-x86_64-linux-gnu.so. Then, you can run DynaHB
 following the instructions in "How to Run an Example"


##**_How to Run an Example_**

1: Install a distributed file system, e.g., NFS, HDFS. Set the shared-directory as "/mnt/data". 
If you just want to run DynaHB on a single-machine, you can just "mkdir /mnt/data" without installing NFS.

2: Partition dynamic graphs by using programs in "python/data_processing/tgnn" and move the result file to the root path of the dataset.

3. Run python/example/dcrnn/dcrnn_example.py with the following arguments for $i^{th}$ worker.
```
--id=i
--worker_num=5
--ifctx=true
```
 
4. We will explain the detailed meanings of the context of DynaHB.
```
'ip': "xxx.xxx.xxx.xxx", ip of the main worker ensuring all workers can communicate by network
'worker_num': number of workers for training
'device': training with CPU or GPU, options [cuda, cpu]
'partitionMethod': partition methods supported by DynaHB, options [hash, metis, load_aware]
'is_adap_batch': if using adaptive RL-based batch adjuster, Boolean type
'is_batch_pool': if using the batch reservior, Boolean type
'dist_mode': the mode for parameter updating, options [asyn, sync]
'capacity_for_bos': if using snapshot batch when training on DynaHB

// the following parameters are for specific datasets, and we will provide a concrete example.
'data_path': "/mnt/data/dataset/ia-slashdot-reply-dir",  //the path of dataset
'feature_dim': 2,  // the dimension size of features
'data_num': 51083,  // the number of vertex
'hidden': [16],  // the hidden size of DGNN models
'class_num': 1,  // the number of labels, 1 is for regression tasks, and others are for classification tasks
'train_ratio': 0.4,  // the ratio of train set
'window_size': 6,  # 6,31,78, batch/train/total  // batch size of snapshots
'batch_size': 510,  # 510,10216,51083, batch/local/total // batch size of vertices
'lr': 0.001, // learning rate

```




