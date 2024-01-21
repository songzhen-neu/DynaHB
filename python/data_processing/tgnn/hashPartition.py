# worker_num = 1
# data_num = 129
# write_path='/mnt/data/dataset/england_covid/nodesPartition.hash'+str(worker_num)+'.txt'

# worker_num = 1
# data_num = 6
# write_path='/mnt/data/dataset/test/nodesPartition.hash'+str(worker_num)+'.txt'

# worker_num = 1
# data_num = 1000
# write_path='/mnt/data/dataset/twitter_tennis/nodesPartition.hash'+str(worker_num)+'.txt'

# worker_num = 5
# data_num = 51083
# write_path='/mnt/data/dataset/ia-slashdot-reply-dir/nodesPartition.hash'+str(worker_num)+'.txt'

# worker_num = 5
# data_num = 24575382
# write_path='/mnt/data/dataset/soc-bitcoin/nodesPartition.hash'+str(worker_num)+'.txt'

# worker_num = 5
# data_num = 2302925
# write_path='/mnt/data/dataset/soc-flickr-growth/nodesPartition.hash'+str(worker_num)+'.txt'

# worker_num = 5
# data_num = 2146057
# write_path='/mnt/data/dataset/rec-amazon-ratings/nodesPartition.hash'+str(worker_num)+'.txt'

# worker_num = 5
# data_num = 3223589
# write_path='/mnt/data/dataset/soc-youtube-growth/nodesPartition.hash'+str(worker_num)+'.txt'

# worker_num = 5
# data_num = 10356390
# write_path='/mnt/data/dataset/rec-amz-Books/nodesPartition.hash'+str(worker_num)+'.txt'

# worker_num = 5
# data_num = 545196
# write_path='/mnt/data/dataset/stackexch/nodesPartition.hash'+str(worker_num)+'.txt'

with open(write_path, 'w') as file:
    write_data = [[] for i in range(worker_num)]
    for i in range(worker_num):
        write_data[i] = [str(num) for num in range(i, data_num, worker_num)]
        write_data[i] = "\t".join(write_data[i])
        write_data[i]+="\n"
        file.write(write_data[i])