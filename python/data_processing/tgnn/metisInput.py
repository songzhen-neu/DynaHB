# from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
# loader = ChickenpoxDatasetLoader()



# from torch_geometric_temporal.dataset import IaSlashdotReplyDirDatasetLoader
# loader = IaSlashdotReplyDirDatasetLoader()
# fileName='/mnt/data/dataset/ia-slashdot-reply-dir'      #chickenpox
# edgesMetisFileName=fileName+'/metis.txt'
# index=97

# from torch_geometric_temporal.dataset import SocBitcoinDatasetLoader
# loader = SocBitcoinDatasetLoader()
# fileName='/mnt/data/dataset/soc-bitcoin'      #chickenpox
# edgesMetisFileName=fileName+'/metis.txt'
# index=97

# from torch_geometric_temporal.dataset import SocYoutubeGrowthDatasetLoader
# loader = SocYoutubeGrowthDatasetLoader()
# fileName='/mnt/data/dataset/soc-youtube-growth'      #chickenpox
# edgesMetisFileName=fileName+'/metis.txt'
# index=97

# from torch_geometric_temporal.dataset import SocFlickrGrowthDatasetLoader
# loader = SocFlickrGrowthDatasetLoader()
# fileName='/mnt/data/dataset/soc-flickr-growth'      #chickenpox
# edgesMetisFileName=fileName+'/metis.txt'
# index=97

# from torch_geometric_temporal.dataset import RecAmazonRatingsDatasetLoader
# loader = RecAmazonRatingsDatasetLoader()
# fileName='/mnt/data/dataset/rec-amazon-ratings'      #chickenpox
# edgesMetisFileName=fileName+'/metis.txt'
# index=97

from torch_geometric_temporal.dataset import RecAmzBooksDatasetLoader
loader = RecAmzBooksDatasetLoader()
fileName='/mnt/data/dataset/rec-amz-Books'      #chickenpox
edgesMetisFileName=fileName+'/metis.txt'
index=97




isDynamic=True



dataset = loader.get_global_dataset(0)


if isDynamic:
    nodeNum = dataset.features[index].shape[0]
    featDim = dataset.features[index].shape[1]
    edgesNum = dataset.edge_weights[index].shape[0]
    edges = dataset.edges[index]
    edgesFeat = dataset.edge_weights[index]
else:
    nodeNum = dataset.features[index].shape[0]
    featDim = dataset.features[index].shape[1]
    edgesNum = dataset.edge_weight.shape[0]
    edges = dataset.edge_index
    edgesFeat = dataset.edge_weight


# print(nodeNum,featDim,edgesNum,edgesFeat)

# fileName='../data/chickenpox/'
# 
# edgesMetisFileName=fileName+'/metis.txt'



if __name__ == '__main__':


    adjs = {}
    count = 0

    for i in range(edgesNum):
        vid=edges[0][i]
        nid=edges[1][i]
        if vid!=nid:
            if adjs.__contains__(vid):
                if not adjs[vid].__contains__(nid):
                    adjs[vid].add(nid)
                    count+=1
            else:
                set_tmp=set()
                adjs[vid]=set_tmp
                adjs[vid].add(nid)
                count += 1
            if adjs.__contains__(nid):
                if not adjs[nid].__contains__(vid):
                    adjs[nid].add(vid)
            else:
                set_tmp=set()
                adjs[nid]=set_tmp
                adjs[nid].add(vid)

    # start to write
    metisFileWrite=open(edgesMetisFileName,'w+')
    metisFileWrite.write(str(nodeNum)+' '+str(count)+'\n')
    for i in range(nodeNum):
        if adjs.__contains__(i):
            neighbStr=''
            for neibor in adjs[i]:
                neighbStr+=str(neibor+1)+' '
            neighbStr=neighbStr[:-1]
            neighbStr+='\n'
            metisFileWrite.write(neighbStr)
        else:
            neighbStr=''
            neighbStr+='\n'
            metisFileWrite.write(neighbStr)
            print("not contains {0}".format(i))
    metisFileWrite.close()


