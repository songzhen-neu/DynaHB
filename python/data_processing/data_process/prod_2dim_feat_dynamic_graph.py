import numpy as np


# path='/mnt/data/dataset/SFHH-conf-sensor/SFHH-conf-sensor.edges'
# window=20
# is_weighted=False
# delimiter=','
# skip_header=0


path='/mnt/data/dataset/ia-slashdot-reply-dir/ia-slashdot-reply-dir.edges'
window=5000
is_weighted=True
delimiter=' '
skip_header=2

# path='/mnt/data/dataset/soc-flickr-growth/soc-flickr-growth.edges'
# window=300
# is_weighted=True
# delimiter=' '
# skip_header=1

def get_src_tgt_wei_time(data):
    if not is_weighted:
        source_vertices = data[:, 0].astype(int)
        target_vertices = data[:, 1].astype(int)
        timestamps = data[:, 2].astype(int)
        return source_vertices,target_vertices,None,timestamps
    else:
        source_vertices = data[:, 0].astype(int)
        target_vertices = data[:, 1].astype(int)
        edge_weights = data[:, 2].astype(int)
        timestamps = data[:, 3].astype(int)
        return source_vertices, target_vertices,edge_weights, timestamps


def get_encoded_src_tgt(source_vertices,target_vertices):
    # encode vertex
    unique_vertices = np.unique(np.concatenate((source_vertices, target_vertices)))
    vertex_num=unique_vertices.size
    vertex_mapping = {old_vertex: new_vertex for new_vertex, old_vertex in enumerate(unique_vertices)}
    source_vertices_mapped = np.vectorize(vertex_mapping.get)(source_vertices)
    target_vertices_mapped = np.vectorize(vertex_mapping.get)(target_vertices)
    return source_vertices_mapped,target_vertices_mapped,vertex_num

def get_snapshot_mask(timestamps):
    # encode time
    unique_timestamps = np.unique(timestamps)
    unique_timestamps = np.sort(unique_timestamps)
    start_id = unique_timestamps[0]
    time_itv = (unique_timestamps[-1] - unique_timestamps[0]) / window

    snap_mask = [None for i in range(window)]
    for i in range(window):
        snap_mask[i] = (start_id + i * time_itv < timestamps) & (timestamps < start_id + (i + 1) * time_itv)
    return snap_mask

if __name__=='__main__':
    data=np.loadtxt(path,skiprows=skip_header)
    source_vertices, target_vertices,edge_weights, timestamps=get_src_tgt_wei_time(data)

    source_vertices_mapped,target_vertices_mapped,vertex_num=get_encoded_src_tgt(source_vertices,target_vertices)
    snap_mask=get_snapshot_mask(timestamps)



    edge_snapshots = [None for i in range(window)]
    edge_weight_snapshots=[None for i in range(window)]
    for i in range(window):
        mask = snap_mask[i]
        edge_snapshots[i] = np.array([source_vertices_mapped[mask], target_vertices_mapped[mask]])
        edge_weight_snapshots[i]=np.array(edge_weights[mask])

    edge_snapshots=[arr for arr in edge_snapshots if arr.size >0]
    edge_weight_snapshots = [arr for arr in edge_weight_snapshots if arr.size > 0]
    snapshot_num=len(edge_snapshots)
    edge_num=len(edge_weights)



    print('a')

