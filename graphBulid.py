'''
Description: 构建快照
Usage:
Parameters:  xxxx-preprocess.txt
Return:
Author: Ying Jie
LastEditTime: 2023-09-17 13:53:46
'''
import os
import networkx as nx
import pickle
import configparser
from utils import create_folder_for_file
graphName = 'Graph'
G = nx.MultiGraph(name=graphName, data=True, align='vertical')  # undirected
GraphSeq = {}

def graphBuild(readFilepath,graph_node_num,writeFillpath):
    graph_id = 0
    n = graph_node_num

    with open(readFilepath) as f:
        while True:
            line = f.readline().strip()
            print(line)
            if not line: break
            splits = line.split(' ')
            subUUID, subType, eventType, objUUID, objType, timeStamp = splits[0], splits[1], splits[2], splits[3], \
            splits[4], splits[5]
            print(eventType, subType, objType)
            if not G.has_node(subUUID):
                G.add_node(subUUID, type=subType, time=timeStamp)
            if not G.has_node(objUUID):
                G.add_node(objUUID, type=objType, time=timeStamp)

            if not G.has_edge(subUUID, objUUID, eventType):
                G.add_edge(subUUID, objUUID, time=timeStamp, key=eventType)
            else:
                G[subUUID][objUUID][eventType]['time'] = timeStamp  # 仅维护同类型事件的最新的一条
            G.nodes()[subUUID]['time'], G.nodes()[objUUID]['time'] = timeStamp, timeStamp  # 更新节点时间

            if len(G.nodes()) >= n:
                GraphSeq[graph_id] = G.copy()
                graph_id += 1
                G.clear()
    ##防止余下数据丢失
    GraphSeq[graph_id] = G.copy()
    graph_id += 1
    G.clear()
    os.makedirs(os.path.dirname(writeFillpath),exist_ok=True)
    # specificSnapshotFile = r"ProcessedData/drapra3/theia/train/graph/graph.pkl"
    # specificSnapshotFile = r"ProcessedData/drapra3/theia/test/data/graph.pkl"
    # create_folder_for_file(specificSnapshotFile)
    # with open(os.getcwd() + os.path.join(SnapshotDir,  '22_test_ext.pkl'), 'wb') as fs:
    with open(writeFillpath, 'wb') as fs:
        pickle.dump(GraphSeq, fs)
    return graph_id
















