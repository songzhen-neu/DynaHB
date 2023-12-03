import json

# 创建一个Python字典（或其他支持的数据结构）

edge_weight = {}
edge_index = {}
time_periods = 4

edge_index['0'] = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 0], [2, 1], [3, 2], [4, 3], [5, 2],[0,0], [1, 1], [2, 2],
                   [3, 3], [4, 4], [5, 5]]
edge_index['1'] = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 0], [2, 1], [3, 2], [4, 3], [5, 3],[0,0], [1, 1], [2, 2],
                   [3, 3], [4, 4], [5, 5]]
edge_index['2'] = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 0], [2, 1], [3, 2], [4, 3], [5, 4],[0,0], [1, 1], [2, 2],
                   [3, 3], [4, 4], [5, 5]]
edge_index['3'] = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 0], [2, 1], [3, 2], [4, 3], [5, 4],[0,0], [1, 1], [2, 2],
                   [3, 3], [4, 4], [5, 5]]

edge_weight['0'] = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,1, 2, 3, 4, 5]
edge_weight['1'] = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,1, 2, 3, 4, 5]
edge_weight['2'] = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,1, 2, 3, 4, 5]
edge_weight['3'] = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1,1, 2, 3, 4, 5]

edge_mapping = {
    "edge_index": edge_index,
    "edge_weight": edge_weight,
}

y = [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]]

dataset = {
    "edge_mapping": edge_mapping,
    "time_periods": time_periods,
    "y": y
}

# data = {
#     "name": "John",
#     "age": 30,
#     "city": "New York"
# }

# 指定要写入的JSON文件的文件名
filename = "/mnt/data/dataset/test/test.json"

# 使用with语句打开文件，确保文件在退出with块时自动关闭
with open(filename, 'w') as json_file:
    # 使用json.dump()函数将数据写入JSON文件
    json.dump(dataset, json_file, indent=4)  # indent参数可选，用于缩进JSON数据，使其更易读

# 文件已经写入，关闭文件
