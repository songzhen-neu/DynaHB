# from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
# loader = ChickenpoxDatasetLoader()
import numpy as np
import pandas

# from torch_geometric_temporal.dataset import IaSlashdotReplyDirDatasetLoader
# loader = IaSlashdotReplyDirDatasetLoader()
# dataset = loader.get_global_dataset(0)
# fileName = '/mnt/data/dataset/ia-slashdot-reply-dir'     #chickenpox
# workerNum = 5
# nodeNum = 51083

# from torch_geometric_temporal.dataset import SocBitcoinDatasetLoader
# loader = SocBitcoinDatasetLoader()
# dataset = loader.get_global_dataset(0)
# fileName = '/mnt/data/dataset/soc-bitcoin'     #chickenpox
# workerNum = 5
# nodeNum = 24575382

# from torch_geometric_temporal.dataset import SocFlickrGrowthDatasetLoader
# loader = SocFlickrGrowthDatasetLoader()
# dataset = loader.get_global_dataset(0)
# fileName = '/mnt/data/dataset/soc-flickr-growth'     #chickenpox
# workerNum = 5
# nodeNum = 2302925

# from torch_geometric_temporal.dataset import RecAmzBooksDatasetLoader
# loader = RecAmzBooksDatasetLoader()
# dataset = loader.get_global_dataset(0)
# fileName = '/mnt/data/dataset/rec-amz-Books'     #chickenpox
# workerNum = 5
# nodeNum = 10356390

# from torch_geometric_temporal.dataset import RecAmazonRatingsDatasetLoader
# loader = RecAmazonRatingsDatasetLoader()
# dataset = loader.get_global_dataset(0)
# fileName = '/mnt/data/dataset/rec-amazon-ratings'     #chickenpox
# workerNum = 5
# nodeNum = 2146057


from torch_geometric_temporal.dataset import SocYoutubeGrowthDatasetLoader
loader = SocYoutubeGrowthDatasetLoader()
dataset = loader.get_global_dataset(0)
fileName = '/mnt/data/dataset/soc-youtube-growth'     #chickenpox
workerNum = 5
nodeNum = 3223589


metisFileName = fileName + '/metis.txt.part.' + str(workerNum)
fileWriteName = fileName + '/nodesPartition'+'.metis'+str(workerNum)+'.txt'

if __name__ == '__main__':
    if workerNum==1:
        metisFileWrite = open(fileWriteName, 'w+')
        for i in range(nodeNum):
            metisFileWrite.write(str(i)+'\t')
        metisFileWrite.close()

    else:
        # metisFileRead = open(metisFileName, 'r')
        metisFileWrite = open(fileWriteName, 'w+')
        # allLines = metisFileRead.readlines()
        allLines=np.array(pandas.read_csv(metisFileName)).flatten()

        for i in range(workerNum):
            node_for_wk_i=np.where(allLines==i)
            result_i='\t'.join(map(str,node_for_wk_i[0]))
            metisFileWrite.write(result_i)

        # nodes = [str() for i in range(workerNum)]
        # nodeId = 0
        # for eachLine in allLines:
        #     eachLine = eachLine[:-1]
        #     workerId = int(eachLine)
        #     nodes[workerId] += str(nodeId) + '\t'
        #     nodeId += 1
        # for i in range(workerNum):
        #     nodes[i] = nodes[i][:-1] + '\n'
        #     metisFileWrite.write(nodes[i])
        metisFileWrite.close()
        # metisFileRead.close()
