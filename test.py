import pickle
import torch.nn as nn
from Loader import *
import torch
import argparse
import random
from utils import *
from config import *
parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')
parser.add_argument('--dataSet', type=str, default='ProcessedData')
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--b_sz', type=int, default=20)
parser.add_argument('--seed', type=int, default=824)
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='./experiments.conf')  #/src
parser.add_argument('--num_layers',type=int,default=2)
parser.add_argument('--hidden_emb_size',type=int,default=128)
args = parser.parse_args()

# def evaluate(groundtruth,anomalous_graph_dist,id_to_uuid):
#     f_gt = open(groundtruth, 'r')
#     # f_alarm = open(alarm, 'r')
#     predict_tp = []
#     predict_fp = []
#     eps = 1e-10
#
#     f = open(id_to_uuid, 'r')
#     node_map = {}
#     ##将id与源相对应。
#     for line in f:
#         line = line.strip('\n').split(' ')
#         node_map[line[1]] = line[0]
#     f.close()
#
#     gt = {}
#     ##存储groundtruth的id,
#     for line in f_gt:
#         # uuid = line.strip('\n').split(' ')[0]
#         # try:
#         #     gt[int(node_map[uuid])] = uuid
#         # except:
#         #     print("groundtruth不存在")
#         gt[int(line.strip('\n').split(' ')[0])] = 1
#     fn_count = len(gt.keys())
#     # new_groundtruth= []
#     # for id in gt:
#     #     new_groundtruth.append(gt[id])
#     # with open('groundtruth_uuid.txt','w') as wf:
#     #     wf.writelines('\n'.join(new_groundtruth))
#     # ans = []
#     ##tn总预测数，fn总的groundtruth
#     for line in anomalous_graph_dist:
#         if line == '\n': continue
#         # if not ':' in line:
#         #     tot_node = int(line.strip('\n'))
#         #     for i in range(tot_node):
#         #         ans.append('tn')
#         #     for i in gt:
#         #         # try:
#         #         #     ans[i] = 'fn'
#         #         # except:
#         #         #     print("节点断连了")
#         #         ans[i] = 'fn'
#         #     continue
#
#         if not ':' in line:
#             tot_node = int(line.strip('\n'))
#
#             # for i in range(tot_node):
#             #     ans.append('tn')
#             continue
#         line = line.strip('\n')
#         a = int(line.split(':')[0])
#         b = line.split(':')[1].strip(' ').split(' ')
#         flag = 0
#         ### 使用两个数组存储预测出来的key，和预测错误的key
#         """
#         predict_tp = []
#         predict_fp = []
#         """
#
#         for i in b:
#             if i == '': continue
#             if int(i) in gt.keys():
#                 # tp_count +=1
#                 # ans[int(i)] = 'tp'
#                 # flag = 1
#                 if int(i) in predict_tp:
#                     continue
#                 else:
#                     predict_tp.append(int(i))
#                 flag = 1
#         if a in gt.keys():
#             # ans[a] = 'tp'
#             # tp_count += 1
#             if a in predict_tp:
#                 continue
#             else:
#                 predict_tp.append(a)
#         else:
#             if flag == 0:
#                 # fp_count += 1
#                 # ans[a] = 'fp'
#                 if a in predict_fp:
#                     print("已存在")
#                 else:
#                     predict_fp.append(a)
#
#     # tn = 0
#     # tp = 0
#     #
#     # fp = 0
#     # for i in ans:
#     #     if i == 'tp': tp += 1
#     #     if i == 'tn': tn += 1
#     #     if i == 'fp': fp += 1
#         # if i == 'fn': fn += 1
#     tp = len(predict_tp)
#     fp = len(predict_fp)
#
#     fn = fn_count - tp
#     tn = tot_node-fn-tp-fp
#     print(tp, fp, tn, fn)
#     precision = tp / (tp + fp + eps)
#     recall = tp / (tp + fn + eps)
#     fscore = 2 * precision * recall / (precision + recall + eps)
#     print('Precision: ', precision)
#     print('Recall: ', recall)
#     print('F-Score: ', fscore)

def getGT_graph(groundtruthfile,graphfile):
    ##获取每个图所含有的节点id
    GT_id = []
    GT_graph = []
    allGraph = {}
    with open(graphfile, 'rb') as f:
        graphSeq = pickle.load(f)
    for index, snapshot in graphSeq.items():
        allGraph[index] = snapshot
    ###读取GT
    with open(groundtruthfile,'r') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            id = line[0]
            GT_id.append(id)
    print(GT_id)
    ###获取每个GT属于哪个graph
    for id in GT_id:
        result  = [graph_index for graph_index, graph_nodes in allGraph.items() if id in graph_nodes]
        for graph in result:
            if graph not in GT_graph:
                GT_graph.append(graph)
    print("恶意节点所在图"+str(GT_graph))
    return GT_id
def testData(modelFilePathDir,node_net_path,process_lable_vec,process_property_vec,file_evaluation_num,evaluation_subject_num,alpha,beta,yam,max_node_threold,node_threold,GT_id):
    criterion_struct = nn.MSELoss(reduction='sum')
    criterion_anomalous = nn.MSELoss(reduction='sum')
    criterion_property = nn.MSELoss(reduction='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(modelFilePathDir + '/AE-with-three-type.model')
    model.eval()

    ##### get the threshold #####
    """
    测试得到每个节点的阈值
    """
    node_loss_dict = []
    graph_loss_dict = []

    anomalous_node = []
    groundtruth_node_loss = []
    malicious_node = []
    ##存储所有节点的损失值
    all_node_loss = {}
    ##存储所有节点的所在图
    node_to_graph = {}
    file_count = 0
    anomalous_count = 0
    while file_count < file_evaluation_num:
        node_net_file = node_net_path + str(file_count) + "_vec.txt"
        train_set = Train_Loader(node_net_file, process_lable_vec, process_property_vec, evaluation_subject_num)
        model.eval()
        count = 0
        total_loss = 0

        for bx, (idx, struct_data, anomalous_data, property_data) in enumerate(train_set):
            struct_data = struct_data.to(device)
            anomalous_data = anomalous_data.to(device)
            property_data = property_data.to(device)
            sample_struct = model(struct_data, 'struct')
            sample_anomalous = model(anomalous_data, 'anomalous')
            sample_property = model(property_data, 'property')
            count += 1
            loss_struct = criterion_struct(struct_data.to(device), sample_struct).item()
            loss_anomalous = criterion_anomalous(anomalous_data.to(device),
                                                 sample_anomalous).item()
            loss_property = criterion_property(property_data.to(device),
                                               sample_property).item()
            loss = alpha * loss_struct + beta * loss_anomalous + yam * loss_property
            all_node_loss[idx] = loss
            if idx not in node_to_graph:
                node_to_graph[idx] = set()

            node_to_graph[idx].add(file_count)

            if loss > max_node_threold:

                if idx not in malicious_node:
                    malicious_node.append(idx)
            if loss > node_threold:
                if idx not in anomalous_node:
                    anomalous_node.append(idx)
                    anomalous_count += 1
            total_loss += loss
            node_loss = str(idx) + " " + str(loss)
            node_loss_dict.append(node_loss)
            if idx in GT_id:
                groundtruth_loss = "节点"+str(idx)+"的loss为："+str(node_loss)+",节点所在的图为："+str(file_count)
                groundtruth_node_loss.append(groundtruth_loss)


        ##判读是否是恶意图中的节点

        # if (anomalous_count/(bx+1))>0.05:
        #     for node in anomalous_node:
        #         if node not in malicious_node:
        #             malicious_node.append(node)
        #     anomalous_node = []
        # else:
        #     anomalous_node = []

        anomalous_count = 0

        # filepath = "node_loss_graph" + str(file_count) + ".txt"
        # with open(filepath, 'w') as wf:
        #     wf.writelines('\n'.join(node_loss_dict))
        #     wf.writelines('\n')
        file_count += 1
        node_loss_dict = []

    # with open("graph_loss.txt", 'w') as wf:
    #     wf.writelines('\n'.join(graph_loss_dict))
    #     wf.writelines('\n')


    ###存储节点的属性值
    subject_file = r"ProcessedData/drapra3/cadets/evaluation/data/subject.txt"
    subject_property = {}
    with open(subject_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                subject_id, subjectPropertity = line.strip().split(": ", 1)
                subject_property[subject_id] = subjectPropertity
            except:
                continue
    with open("anomalous_node.txt", 'w') as wf:
        for node in malicious_node:
            if node in all_node_loss.keys() and node in node_to_graph.keys() and node in subject_property.keys():
                lines = str(node)+"的损失值为："+str(all_node_loss[node])+","+str(node)+"的属性值为："+subject_property[node]
                wf.writelines(lines)
                wf.writelines('\n')
    # with open("groundtruth_node_loss.txt", 'w') as wf:
    #     wf.writelines('\n'.join(groundtruth_node_loss))
    #     wf.writelines('\n')




    ##生成alarm文件
    # fw = open(alarm_file, 'w')
    ##count_total：记录了共测试了多少个节点


    # fw.write(str(count_total) + '\n')
    # for key in anomaly_node_neibor:
    #     fw.write('\n')
    #     fw.write(str(key) + ':')
    #     neibors = anomaly_node_neibor[key]
    #     for neibor in neibors:
    #         fw.write(' ' +str(neibor))

    #将图的节点写入
    # for data in anomaly_graph_to_node_dist:
    #     fw.write('\n')
    #     fw.write(str(data)+":")
    #
    # fw.close()



    # for i in anomaly_node_dist:
    #     fw.write('\n')
    #     fw.write(str(i) + ':')
    #     ###先仅仅只测试节点
    #     neibor = set()
    #     if i in test_set.adjlist.keys():
    #         for j in test_set.adjlist[i]:
    #             neibor.add(j)
    #             if not j in test_set.adjlist.keys(): continue
    #             for k in test_set.adjlist[j]:
    #                 neibor.add(k)
    #     for j in neibor:
    #         fw.write(' ' + str(j))
    #
    # fw.close()



