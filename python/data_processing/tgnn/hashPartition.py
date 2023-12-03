# worker_num = 1
# data_num = 129
# write_path='/mnt/data/dataset/england_covid/nodesPartition.hash'+str(worker_num)+'.txt'

worker_num = 1
data_num = 6
write_path='/mnt/data/dataset/test/nodesPartition.hash'+str(worker_num)+'.txt'

with open(write_path, 'w') as file:
    write_data = [[] for i in range(worker_num)]
    for i in range(worker_num):
        write_data[i] = [str(num) for num in range(i, data_num, worker_num)]
        write_data[i] = "\t".join(write_data[i])
        write_data[i]+="\n"
        file.write(write_data[i])