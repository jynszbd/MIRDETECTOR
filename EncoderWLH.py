import ast
import os
import re
import pickle
import numpy as np
import configparser
import networkx as nx
from typing import List
# import matplotlib.pyplot as plt
from datetime import datetime
import json
from karateclub.estimator import Estimator
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from WLH import WeisfeilerLehmanHashing
from utils import create_folder_for_file
# from karateclub.utils.treefeatures import WeisfeilerLehmanHashing

d = 5

D = 64

encoderModel = 'AutoEncoder'

class WLEmbedding(Estimator):
    """
    Args:
        wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 2.(Replaced by d)(×)
        d (int): Number of d in papaer(PROGRAPHER).
        attributed (bool): Presence of graph attributes. Default is False. (Replaced by RSGeneration)(×)
        dimensions (int): Dimensionality of embedding. Default is 256.
        workers (int): Number of cores. Default is 4.
        down_sampling (float): Down sampling frequency. Default is 0.0001.
        negative (int): Number of negative samples.
        epochs (int): Number of epochs. Default is 10.
        learning_rate (float): HogWild! learning rate. Default is 0.025.
        min_count (int): Minimal count of graph feature occurrences. Default is 1.
        seed (int): Random seed for the model. Default is 42. (×)
        erase_base_features (bool): Erasing the base features. Default is False. (×)
    """

    def __init__(
            self,
            wl_iterations: int = 1,
            d: int = 3,
            attributed: bool = False,
            dimensions: int = 128,
            workers: int = 8,
            # down_sampling: float = 0.0001,
            down_sampling: float = 0.0,
            epochs: int = 100,
            negative: int = 15,
            learning_rate: float = 0.025,
            min_count: int = 0,
            seed: int = 42,
            erase_base_features: bool = False,
            cache=None,
    ):
        self.wl_iterations = wl_iterations
        self.d = d
        self.attributed = attributed
        self.dimensions = dimensions
        self.workers = workers
        self.negative = negative
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.erase_base_features = erase_base_features
        self.cache = cache

    def fit(self, graphs):
        """
        Fitting a Graph2Vec model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()

        documents = [
            WeisfeilerLehmanHashing(
                (index, graph), self.wl_iterations, self.d, self.attributed, self.erase_base_features, self.cache
            )
            for index, graph in graphs.items()
        ]
        documents_node= []
        for graph_index,graph in enumerate(documents):
            for node,feature in graph.get_graph_features():
                documents_node.append(TaggedDocument(words=feature,tags=[str(node)]))



        self.model = Doc2Vec(
            documents_node,
            vector_size=self.dimensions,
            window=2,  # window = 1 / fr - 1
            min_count=self.min_count,
            dm=1,  # 'distributed memory' (PV-DM), 考虑上下文，如果用PV-DBOW就不考虑位置
            sample=self.down_sampling,  # 对高频词进行随机降采样
            workers=self.workers,
            epochs=self.epochs,
            negative=self.negative,  # 15个负样本
            alpha=self.learning_rate,
            seed=self.seed,
        )
        embedding_and_tag = {}
        for doc in documents_node:
            tag = doc.tags[0]
            try:
                embedding_node = self.model.docvecs[tag]
                embedding_and_tag[tag] = embedding_node
            except:
                continue
        # self._embedding = [self.model.dv[str(i)] for i, _ in enumerate(documents_node)]
        self._embedding = embedding_and_tag

    def get_embedding(self) -> dict:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        # return np.array(self._embedding)
        return self._embedding
    def infer(self, graphs, model) ->dict:
        """Infer the graph embeddings.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        self._set_seed()

        node_embeddings = {}
        wfh = WeisfeilerLehmanHashing((0, graphs),self.wl_iterations,self.d,self.attributed,self.erase_base_features,self.cache)
        graph_features = wfh.get_graph_features()

        for node, feature in graph_features:
            doc_words = []
            parts = feature.split(',')
            for part in parts:
                clean_part = part.strip().replace('(','').replace(')','')
                if clean_part != '':
                    doc_words.append(str(clean_part))
            node_embedding = model.infer_vector(doc_words,alpha=self.learning_rate,min_alpha=0.00001, epochs=self.epochs)
            node_embeddings[node] = node_embedding


        # graphs = self._check_graphs(graphs)

        # documents = [doc.get_graph_features() for _, doc in enumerate(documents)]
        #
        # embedding = np.array(
        #     [
        #         model.infer_vector(
        #             doc, alpha=self.learning_rate, min_alpha=0.00001, epochs=self.epochs
        #         )
        #         for doc in documents
        #     ]
        # )

        return node_embeddings

def read(json_file):
    data_ = open(json_file, "r", encoding="utf-8")
    docs = json.load(data_)
    return docs

"""-----------------train with all pkl file ------------------"""
def train_with_graph(filepath,embeddingfile):

    allGraph = {}
    with open(filepath, 'rb') as f:
        graphSeq = pickle.load(f)
    for index, graph in graphSeq.items():
        allGraph[index] = graph
    current_time = datetime.now()
    print("开始时间为：",current_time)
    print("快照提取结束")
    WLEmbedding_vec = WLEmbedding()
    WLEmbedding_vec.fit(allGraph)
    print("WL特征嵌入")
    # modelpath = r"ProcessedData/drapra3/theia/WLModel/WLModel"
    modelpath = r"ProcessedData/drapra3/cadets/WLModel/WLModel_d_3_128"
    create_folder_for_file(modelpath)
    WLEmbedding_vec.model.save(modelpath)
    print("model save")
    model = Doc2Vec.load(modelpath)

    create_folder_for_file(embeddingfile)
    current_time = datetime.now()
    print("结束时间为：", current_time)
    embedding = WLEmbedding_vec.get_embedding()
    ##输出对应的embedding。
    print("输出embedding")
    WL_embed = []
    for key in embedding.keys():
        # node_to_embedding = str(key)+" "+str(embedding[key])
        node_to_embedding = str(key) + " " + " ".join(str(x) for x in embedding[key])
        WL_embed.append(node_to_embedding)

    with open (embeddingfile , 'w') as wf:
        wf.writelines('\n'.join(WL_embed))
        wf.writelines('\n')
    print("embedding 输出结束")


"""---------输出以图形式的train数据的图 -----------"""
"""在使用整体的VAE进行训练时，是不用变的，用的是全部的WL_vec.txt文件。在测试时，需要用分别的WL_num_vec.txt
        测试每个图的阈值和每个节点的阈值
"""
def write_all_train_graph(filepath,embeddingpath):


    WL_embed = []
    allGraph = {}

    with open(filepath, 'rb') as f:
        graphSeq = pickle.load(f)
    for index, snapshot in graphSeq.items():
        allGraph[index] = snapshot
    print("快照提取结束")
    WLEmbedding_vec = WLEmbedding()
    # modelpath = r"ProcessedData/drapra3/theia/WLModel/WLModel"
    modelpath = r"ProcessedData/drapra3/cadets/WLModel/WLModel_d_3_128"
    model = Doc2Vec.load(modelpath)
    WL_all_embedding = []
    create_folder_for_file(embeddingpath)
    for index in allGraph:
        embeddings = WLEmbedding_vec.infer(graphs=allGraph[index], model=model)
        for key in embeddings.keys():
            # node_to_embedding = str(key)+" "+str(embedding[key])
            node_to_embedding = str(key) + " " + " ".join(str(x) for x in embeddings[key])
            WL_embed.append(node_to_embedding)
            # WL_all_embedding.append(node_to_embedding)
        filepath = embeddingpath + str(index)+"_vec.txt"
        with open (filepath , 'w') as wf:
            wf.writelines('\n'.join(WL_embed))
            wf.writelines('\n')
        print("第"+str(index)+"个图的embedding 输出结束")
        WL_embed = []


    # filepath = embeddingpath + "vec.txt"
    # with open(filepath, 'w') as wf:
    #     wf.writelines('\n'.join(WL_all_embedding))
    #     wf.writelines('\n')
    # print("embedding 输出结束")


"""-----test------"""
def write_test_graph(filepath,embeddingpath):
    WL_embed = []
    allGraph = {}

    with open(filepath, 'rb') as f:
        graphSeq = pickle.load(f)
    for index, snapshot in graphSeq.items():
        allGraph[index] = snapshot
    print("快照提取结束")
    WLEmbedding_vec = WLEmbedding()
    modelpath = r"ProcessedData/drapra3/cadets/WLModel/WLModel_d_3_128"
    model = Doc2Vec.load(modelpath)

    create_folder_for_file(embeddingpath)
    count = 0
    for index in allGraph:
        embeddings = WLEmbedding_vec.infer(graphs=allGraph[index], model=model)
        for key in embeddings.keys():
            # node_to_embedding = str(key)+" "+str(embedding[key])
            node_to_embedding = str(key) + " " + " ".join(str(x) for x in embeddings[key])
            WL_embed.append(node_to_embedding)
        filepath = embeddingpath + str(index)+"_vec.txt"
        with open (filepath , 'w') as wf:
            wf.writelines('\n'.join(WL_embed))
            wf.writelines('\n')
        print("第"+str(index)+"个图的embedding 输出结束")
        WL_embed = []
        count += 1

    return count
