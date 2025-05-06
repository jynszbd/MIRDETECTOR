import hashlib
import networkx as nx
from typing import List, Dict, Tuple
from collections import OrderedDict


class WeisfeilerLehmanHashing(object):
    """
    Weisfeiler-Lehman feature extractor class.

    Args:
        graph (NetworkX graph): NetworkX graph for which we do WL hashing.
        wl_iterations (int): Number of WL iterations.
        attributed (bool): Presence of attributes.
        erase_base_feature (bool): Deleting the base features.
    """

    def __init__(
            self,
            graph: nx.classes.graph.Graph,
            wl_iterations: int,
            d: int,
            attributed: bool,
            erase_base_features: bool,
            cache=None,
    ):
        """
        Initialization method which also executes feature extraction.
        """
        self.wl_iterations = wl_iterations
        self.d = d
        self.cache = cache
        self.graph = graph
        self.features = {}
        self.extracted_features = {}
        self.attributed = attributed
        self.erase_base_features = erase_base_features
        if self.cache:
            self._set_features_cache()
        else:
            self._set_features()
        # self._do_recursions()

    def _set_features(self):
        self.cache = {}
        index, G = self.graph
        self.cache[index] = {}
        for v in G.nodes():
            self.cache[index][v] = {}
            for d in range(self.d + 1):
                self.cache[index][v][d] = {}
        for v in G.nodes():
            self.RSGeneration(index, G, self.d, v)

        index = self.graph[0]
        for node in self.graph[1].nodes():
            ##生成每个节点的邻居结构字符串
            feature = ",".join([str(i) for i in self.cache[index][node][self.d]])
            # hash_object = hashlib.md5(feature.encode())
            # hashing = hash_object.hexdigest()
            self.features[node] = [feature, int(self.graph[1].nodes[node]['time'])]
        self.extracted_features = {k: [str(v[0]), v[1]] for k, v in self.features.items()}
        ## self.extracted_features 保存的是节点的id和相对应的hash值以及time
    def _set_features_cache(self):
        """
        Creating the features.
        """
        index = self.graph[0]

        for node in self.graph[1].nodes():
            feature = ",".join([str(i) for i in self.cache[index][node][self.d]])
            hash_object = hashlib.md5(feature.encode())
            hashing = hash_object.hexdigest()
            self.features[node] = [hashing, int(self.graph[1].nodes[node]['time'])]
        self.extracted_features = {k: [str(v[0]), v[1]] for k, v in self.features.items()}

    def _erase_base_features(self):
        """
        Erasing the base features
        """
        for k, v in self.extracted_features.items():
            del self.extracted_features[k][0]

    def _do_a_recursion(self):
        """
        The method does a single WL recursion.

        Return types:
            * **new_features** *(dict of strings)* - The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.graph.nodes():
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = {
            k: self.extracted_features[k] + [v] for k, v in new_features.items()
        }
        return new_features

    def _do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.wl_iterations):
            self.features = self._do_a_recursion()
        if self.erase_base_features:
            self._erase_base_features()

    def get_node_features(self) -> Dict[int, List[str]]:
        """
        Return the node level features.
        """
        return self.extracted_features

    def get_graph_features(self) -> List[Tuple[int,str]]:
        """
        Return the graph level features.
        """
        # return [
        #     features[0]
        #     for node, features in sorted(self.extracted_features.items(), key=lambda x: x[1][1])
        # ]
        return [
            (node,features[0])
            for node, features in sorted(self.extracted_features.items(), key=lambda x: x[1][1])
        ]
    # def get_tags(self) -> List[str]:
    #     """
    #     Return the graph level features.
    #     """
    #     # return [
    #     #     features[0]
    #     #     for node, features in sorted(self.extracted_features.items(), key=lambda x: x[1][1])
    #     # ]
    #     return [
    #         node
    #         for node, features in sorted(self.extracted_features.items(), key=lambda x: x[1][1])
    #     ]
    def _SortAndDeduplicate(self, L):
        """
        sort and de-duplicate List L [[type, time],[type, time],...]
        """
        R = []
        L = sorted(L, key=lambda t: t[1])
        keyL = dict(L).keys()
        for k in keyL:
            for l in reversed(L):
                if l[0] == k:
                    R.append(l)
                    break
        return R

    def RSGeneration(self, index, G, d, v):  # G:graph, d:hop number, v:centric node
        R = ()

        if self.cache.get(index).get(v).get(d):
            return self.cache[index][v][d]
        if d == 0:
            R = (*R, list(G.nodes[v].values())[0])  # (entityType)
            self.cache[index][v][d] = R
        else:
            F, I = (), ()
            for u in G.neighbors(v):
                F = (*F, self.flatten_tuple(self.RSGeneration(index, G, d - 1, u)))
                # F.append(RSGeneration(G, d-1, u)[0])
                for key, _ in dict(G[v][u]).items(): I = (*I, key)  # [(eventType),...]

            F = tuple(sorted((set(F))))  # 先set再sort会好点？
            I = tuple(sorted((set(I))))

            R = (*R, self.flatten_tuple(self.RSGeneration(index, G, d - 1, v)))
            for element in F: R = (*R, element)
            for element in I: R = (*R, element)
            self.cache[index][v][d] = R
        return R

    def flatten_tuple(self, nested_tuple):
        flattened_list = []
        for item in nested_tuple:
            if isinstance(item, tuple):  # 如果当前元素是元组，则递归调用flatten_tuple函数
                flattened_list.extend(self.flatten_tuple(item))
            else:
                flattened_list.append(item)  # 如果当前元素不是元组，则将其添加到结果列表中
        return tuple(flattened_list)


