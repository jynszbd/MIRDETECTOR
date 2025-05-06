"""
以图的形式训练VAE模型，获取每个图的阈值和每个节点的阈值
"""
import torch.nn as nn
from model import VariationalAutoencoder
from Loader import Train_Loader
from datetime import timedelta
import time
import torch
import argparse
import random
from collections import defaultdict

from utils import *
from config import *

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')
parser.add_argument('--dataSet', type=str, default='ProcessedData')
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=150)
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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""训练时，用的是整体的数据，测试时，用的是分图的，便于得到阈值"""
def train(struct_size,anomalous_size,property_size,subject_num,alpha,beta,yam):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # load ProcessedData
    print('start to train VAE')

    batch_size = 32
    lr = 0.0001  # learning rate
    w_d = 1e-5  # weight decay
    momentum = 0.9
    node_net_file = wl_train_all_file
    processed_property_vec = processed_train_dir + "subject_property_vector.txt"
    process_lable_vec = processed_train_vec + "processlabelVec.txt"
    train_set = Train_Loader(node_net_file,process_lable_vec,processed_property_vec,subject_num)

    train_ = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True
    )

    metrics = defaultdict(list)
    model = VariationalAutoencoder(latent_struct_dims=int(struct_size/4),latent_anomalous_dim=int(anomalous_size/2),latent_property_dim = int(property_size/2),struct_size=struct_size,anomalous_size=anomalous_size,property_size=property_size)
    model.to(device)
    criterion_struct = nn.MSELoss(reduction='sum')
    criterion_anomalous = nn.MSELoss(reduction='sum')
    criterion_property = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_d,eps=1e-8)
    model.train()
    min_epoch_loss = 100
    start = time.time()
    count = 0
    for epoch in range(args.epochs):
        ep_start = time.time()
        running_loss = 0.0
        for bx, (idx,struct_data,anomalous_data,property_data) in enumerate(train_):
            struct_data = struct_data.to(device)
            anomalous_data = anomalous_data.to(device)
            property_data = property_data.to(device)
            sample_struct = model(struct_data,'struct')
            sample_anomalous = model(anomalous_data,'anomalous')
            sample_property = model(property_data,'property')
            loss_struct = criterion_struct(struct_data.to(device), sample_struct) + model.encoder_struct.kl
            loss_anomalous = criterion_anomalous(anomalous_data.to(device),sample_anomalous)+model.encoder_anomalous.kl
            loss_property = criterion_property(property_data.to(device),sample_property)+model.encoder_property.kl
            loss = loss_struct*alpha+loss_anomalous*beta+loss_property*yam
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_set)
        metrics['train_loss'].append(epoch_loss)
        ep_end = time.time()
        print('-----------------------------------------------')
        print('[EPOCH] {}/{}\n[LOSS] {}'.format(epoch + 1, args.epochs, epoch_loss))
        print('Epoch Complete in {}'.format(timedelta(seconds=ep_end - ep_start)))
        if epoch_loss < min_epoch_loss:
            torch.save(model, modelFilePathDir + '/AE-with-three-type.model')
            min_epoch_loss = epoch_loss
        ###当epoch的损失连续五个epoch不再下降时，跳出训练
        # if epoch_loss > min_epoch_loss:
        #     count += 1
        #     if count == 5:
        #         break
        # else:
        #     count = 0

    end = time.time()
    print('-----------------------------------------------')
    print('[System Complete: {}]'.format(timedelta(seconds=end -start)))

def getThreold(node_net_path,process_lable_vec,process_property_vec,file_num,subject_num,alpha,beta,yam):
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
    loss_dist = []
    ##total_loss_dict记录的是每个图的总损失，loss_dict记录的是每个节点的损失
    total_loss_dict = []

    file_count = 0


    while file_count <  file_num:
        node_net_file = node_net_path + str(file_count) + "_vec.txt"
        train_set = Train_Loader(node_net_file, process_lable_vec,process_property_vec,subject_num)
        model.eval()
        count = 0
        total_loss = 0
        for bx, (idx,struct_data,anomalous_data,property_data) in enumerate(train_set):
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
            loss = alpha*loss_struct+beta*loss_anomalous+yam*loss_property
            total_loss += loss
            loss_dist.append(loss)
            node_loss = str(idx) + " " +str(loss)
            node_loss_dict.append(node_loss)
        total_loss_dict.append(total_loss/(bx+1))
        graph_loss = "图"+str(file_count)+"的损失为：" + str(total_loss/(bx+1))
        graph_loss_dict.append(graph_loss)
        filepath = "test_node_loss_graph" + str(file_count) + ".txt"
        with open(filepath, 'w') as wf:
            wf.writelines('\n'.join(node_loss_dict))
            wf.writelines('\n')
        file_count += 1
        node_loss_dict = []
    max_threold = max(loss_dist)
    min_threold = min(loss_dist)

    with open("test_graph_loss.txt",'w') as wf:
        wf.writelines('\n'.join(graph_loss_dict))
        wf.writelines('\n')
    max_graph_threold = max(total_loss_dict)
    min_graph_threold = min(total_loss_dict)
    print("最大节点损失为："+str(max_threold))
    print("最小节点损失为：" + str(min_threold))
    print("最大图损失"+str(max_graph_threold))
    print("最小图损失" + str(min_graph_threold))

    ###标准差
    anomaly_std = np.std(np.array(loss_dist))
    ###平均值
    anomaly_mean = np.mean(np.array(loss_dist))
    ##
    anomaly_cutoff = np.percentile(loss_dist, 90)
    print('节点的平均值为: ', anomaly_mean)
    print('节点的平均值的1.5倍为: ', anomaly_mean*1.5)
    print('节点的平均值的1.5倍加标准差为: ', anomaly_mean * 1.5+anomaly_std)
    print('节点的90分位数为: ', np.percentile(loss_dist, 90))
    print('节点的80分位数为: ', np.percentile(loss_dist, 80))
    print('节点的70分位数为: ', np.percentile(loss_dist, 70))
    print('节点的60分位数为: ', np.percentile(loss_dist, 60))

    print('图的90分位数为: ', np.percentile(total_loss_dict, 90))
    print('图的80分位数为: ', np.percentile(total_loss_dict, 80))
    print('图的70分位数为: ', np.percentile(total_loss_dict, 70))
    print('图的60分位数为: ', np.percentile(total_loss_dict, 60))
    malice_node = anomaly_mean*1.9+anomaly_std
    anomalous_node = anomaly_cutoff
    return max_threold,malice_node
    # print(anomaly_mean, anomaly_std)
    # print('anomaly threshold: ', anomaly_cutoff)
    # print("count" + str(count))
    # print("max_threold" + str(max_threold))
    # print("min_threold" + str(min_threold))








