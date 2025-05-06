import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Loader(torch.utils.data.Dataset):
    def __init__(self):
        super(Loader, self).__init__()
        self.dataset_struct = {}
        self.dataset_anomalous = {}
        self.data_property = {}
        self.idx2processnum = []

    def __len__(self):
        return len(self.dataset_struct)

    def __getitem__(self, idx):
        data_struct = self.dataset_struct[self.idx2processnum[idx]]
        data_anomalous = self.dataset_anomalous[self.idx2processnum[idx]]
        data_property = self.data_property[self.idx2processnum[idx]]
        return self.idx2processnum[idx],data_struct,data_anomalous,data_property


class Train_Loader(Loader):
    def __init__(self, node_net_file,process_lable_vec,process_property_vec,subject_num):
        super(Train_Loader, self).__init__()
        data_stuct = []
        label_data = []
        property_data = []
        node_map = {}  # map node to Node_ID
        process_vec = {}
        label_map = {}  # map label to Label_ID
        index = 0
        with open(node_net_file) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                ###将节点又进行了新的映射
                if int(info[0]) <= subject_num:
                    node_map[info[0]] = index # 节点的index
                    data_stuct.append([float(x) for x in info[1:]])
                    index += 1
        with open(process_property_vec) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                property_data.append([float(x) for x in info[1:]])
                ###将节点又进行了新的映射
                process_vec[info[0]] = property_data
                property_data = []

        array_size = len(data_stuct)
        data_property = np.zeros((array_size,16))
        for keys in node_map:
            ##keys中是原节点的信息
            if keys in process_vec.keys():
                # data_anomalous[node_map[keys]] = label_map[keys]
                ##向量归一化
                scaler = MinMaxScaler(feature_range=(0, 1))
                feature_data = np.array(process_vec[keys])
                feature_data = scaler.fit_transform(feature_data.reshape(-1, 1)).reshape(-1)
                data_property[int(node_map[keys]), :] = feature_data

        data_anomalous = np.zeros((array_size, 15))

        with open(process_lable_vec) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                label_data.append([float(x) for x in info[14:]])
                ###将节点又进行了新的映射
                label_map[info[0]] = label_data
                label_data = []
        ###将process的label按序排列好与node的struct一一对应
        for keys in node_map:
            ##keys中是原节点的信息
            if keys in label_map.keys():
                # data_anomalous[node_map[keys]] = label_map[keys]
                ##向量归一化
                scaler = MinMaxScaler(feature_range=(0, 1))
                feature_data = np.array(label_map[keys])
                feature_data = scaler.fit_transform(feature_data.reshape(-1, 1)).reshape(-1)
                data_anomalous[int(node_map[keys]), :] = feature_data

        data_stuct = np.asarray(data_stuct)
        data_anomalous = np.asarray(data_anomalous)
        data_property = np.asarray(data_property)
        j = 0
        for i in node_map:
            self.idx2processnum += [i]
            # self.dataset_struct[i] = torch.FloatTensor(data_stuct[j])
            ###PCA
            self.dataset_struct[i] = torch.FloatTensor(data_stuct[j])
            self.dataset_anomalous[i] = torch.FloatTensor(data_anomalous[j])
            self.data_property[i] = torch.FloatTensor(data_property[j])
            j += 1
class Test_Loader(Loader):
    def __init__(self, node_net_file,process_lable_vec,process_property_vec,subject_num):
        super(Test_Loader, self).__init__()
        data_stuct = []
        label_data = []
        node_map = {}  # map node to Node_ID
        label_map = {}  # map label to Label_ID
        process_vec = {}
        index = 0
        with open(node_net_file) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                ###将节点又进行了新的映射
                if int(info[0]) <= subject_num:
                    node_map[info[0]] = index  # 节点的index
                    data_stuct.append([float(x) for x in info[1:]])
                    index += 1
        array_size = len(data_stuct)
        data_anomalous = np.zeros((array_size, 15))

        with open(process_lable_vec) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                label_data.append([float(x) for x in info[14:]])
                ###将节点又进行了新的映射
                label_map[info[0]] = label_data
                label_data = []
        with open(process_property_vec) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                label_data.append([float(x) for x in info[1:]])
                ###将节点又进行了新的映射
                process_vec[info[0]] = label_data
                label_data = []

        for keys in node_map:
            ##keys中是原节点的信息
            if keys in label_map.keys():
                # data_anomalous[node_map[keys]] = label_map[keys]
                ##向量归一化
                scaler = MinMaxScaler(feature_range=(0, 1))
                feature_data = np.array(label_map[keys])
                feature_data = scaler.fit_transform(feature_data.reshape(-1, 1)).reshape(-1)
                data_anomalous[int(node_map[keys]), :] = feature_data

        # data_stuct = np.asarray(data_stuct)
        for keys in node_map:
            ##keys中是原节点的信息
            if keys in process_vec.keys():
                process_vec[keys] = np.array(process_vec[keys])
                process_vec[keys] = process_vec[keys].flatten()
                scaler = MinMaxScaler(feature_range=(0, 1))
                label_normalized = scaler.fit_transform(process_vec[keys].reshape(-1, 1)).reshape(-1)
                node_feat_data = np.array(data_stuct[node_map[keys]])
                node_normalized = scaler.fit_transform(node_feat_data.reshape(-1, 1)).reshape(-1)
                new_vector = np.hstack((node_normalized, label_normalized))
                # new_vector = np.concatenate((feat_data[node_map[keys]], label_map[keys]))
                data_stuct[node_map[keys]] = new_vector

        # data_anomalous = np.asarray(data_anomalous)

        # adj_lists = {}
        # ###adj_lists:原数组中的字符
        # with open(event_file) as fp:
        #     for i, line in enumerate(fp):
        #         info = line.strip().split()
        #         assert len(info) == 6
        #         try:
        #             # paper1 = node_map[info[0]]
        #             # paper2 = node_map[info[3]]
        #             paper1 = info[0]
        #             paper2 = info[3]
        #         except:
        #             continue
        #         if paper1 not in adj_lists:
        #             adj_lists[paper1] = set()
        #         adj_lists[paper1].add(paper2)
        #         if paper2 not in adj_lists:
        #             adj_lists[paper2] = set()
        #         adj_lists[paper2].add(paper1)
        # self.adjlist = adj_lists

        j = 0
        for i in node_map:
            self.idx2processnum += [i]
            self.dataset_struct[i] = torch.FloatTensor(data_stuct[j])
            self.dataset_anomalous[i] = torch.FloatTensor(data_anomalous[j])
            j += 1