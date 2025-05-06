import torch.nn as nn
import torch.nn.functional as F
import torch

class VariationalEncoder_Struct(nn.Module):
    def __init__(self, latent_dims, embed_size):
        super(VariationalEncoder_Struct, self).__init__()
        ###三层，到128维的到16
        # self.linear1 = nn.Linear(embed_size, int(embed_size / 2))
        # self.linear2 = nn.Linear(int(embed_size / 2), latent_dims)
        # self.linear5 = nn.Linear(latent_dims, int(latent_dims / 2))
        # self.linear3 = nn.Linear(int(latent_dims / 2), int(latent_dims / 2))
        # self.linear4 = nn.Linear(int(latent_dims / 2), int(latent_dims / 2))
        ###四层，128维的到8
        self.linear1 = nn.Linear(embed_size, int(embed_size / 2))
        self.linear2 = nn.Linear(int(embed_size / 2), latent_dims)
        self.linear5 = nn.Linear(latent_dims, int(latent_dims / 2))
        self.linear6 = nn.Linear(int(latent_dims/2),int(latent_dims/4))
        self.linear3 = nn.Linear(int(latent_dims / 4), int(latent_dims / 4))
        self.linear4 = nn.Linear(int(latent_dims / 4), int(latent_dims / 4))
        ##64维 两层到16
        # self.linear1 = nn.Linear(embed_size, int(embed_size / 2))
        # self.linear2 = nn.Linear(int(embed_size / 2), int(latent_dims / 2))
        # ####两层，到16停止
        # self.linear3 = nn.Linear(int(latent_dims / 2), int(latent_dims / 2))
        # self.linear4 = nn.Linear(int(latent_dims / 2), int(latent_dims / 2))
        ###三层，到8停止
        # self.linear5 = nn.Linear(int(latent_dims / 2), int(latent_dims / 4))
        # self.linear3 = nn.Linear(int(latent_dims / 4), int(latent_dims / 4))
        # self.linear4 = nn.Linear(int(latent_dims / 4), int(latent_dims / 4))
        self.N = torch.distributions.Normal(0, 1)
        print("struct编码器维度:"+str(embed_size)+"中间层维度"+str(int(latent_dims/2)))
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            # 如果CUDA可用，则将均值和标准差移动到CUDA设备上
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
        else:
            # 如果CUDA不可用，则将均值和标准差移动到CPU上
            self.N.loc = self.N.loc.cpu()
            self.N.scale = self.N.scale.cpu()

        # self.N.loc = self.N.loc.cuda()
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        ###三层，128维到16
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear5(x))

        # mu = self.linear3(x)
        # sigma = torch.exp(self.linear4(x))
        ###四层，128维到8
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        #
        ###64维，两层，到16
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        ###到8停止，三层
        # x = F.relu(self.linear5(x))
        mu = self.linear3(x)
        sigma = torch.exp(self.linear4(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z
class Decoder_Struct(nn.Module):
    def __init__(self, latent_dims, embed_size):
        super(Decoder_Struct, self).__init__()
        ###三层，128维到16
        # self.linear1 = nn.Linear(int(latent_dims / 2), latent_dims)
        # self.linear2 = nn.Linear(latent_dims, latent_dims * 2)
        # self.linear3 = nn.Linear(int(embed_size / 2), embed_size)
        # self.linear4 = nn.Linear(embed_size,embed_size)
        ###四层，128维到8维
        self.linear1 = nn.Linear(int(latent_dims / 4), int(latent_dims/2))
        self.linear2 = nn.Linear(int(latent_dims/2),latent_dims)
        self.linear3 = nn.Linear(latent_dims, latent_dims * 2)
        self.linear4 = nn.Linear(int(embed_size / 2), embed_size)
        self.linear5 = nn.Linear(embed_size, embed_size)
        ###64，两层，到16停止
        # self.linear1 = nn.Linear(int(latent_dims / 2), latent_dims)
        # self.linear2 = nn.Linear(latent_dims, embed_size)
        # self.linear3 = nn.Linear(embed_size, embed_size)
        ###64,三层，到8停止
        # self.linear1 = nn.Linear(int(latent_dims / 4), int(latent_dims / 2))
        # self.linear2 = nn.Linear(int(latent_dims / 2), latent_dims)
        # self.linear3 = nn.Linear(latent_dims, embed_size)
        # self.linear4 = nn.Linear(embed_size, embed_size)
    def forward(self, z):
        ###三层，128维到16维
        # z = F.relu(self.linear1(z))
        # z = F.relu(self.linear2(z))
        # z = F.relu(self.linear3(z))
        # z = self.linear4(z)
        ###四层，128维到8维
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = F.relu(self.linear4(z))
        z = self.linear5(z)
        ###64，两层，到16停止
        # z = F.relu(self.linear1(z))
        # z = F.relu(self.linear2(z))
        # z = self.linear3(z)
        ###64,三层，到8停止
        # z = F.relu(self.linear1(z))
        # z = F.relu(self.linear2(z))
        # z = F.relu(self.linear3(z))
        # z = self.linear4(z)
        return z
class VariationalEncoder_Anomalous(nn.Module):
    def __init__(self,latent_dims,embed_size):
        super(VariationalEncoder_Anomalous,self).__init__()
        self.linear1 = nn.Linear(embed_size,int(embed_size/2))
        self.linear2 = nn.Linear(int(embed_size/2),int(embed_size/2))
        self.linear3 = nn.Linear(int(embed_size/2),latent_dims)
        self.linear4 = nn.Linear(int(embed_size/2),latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        print("label编码器维度:" + str(embed_size) + "中间层维度" + str(latent_dims))
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            # 如果CUDA可用，则将均值和标准差移动到CUDA设备上
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
        else:
            # 如果CUDA不可用，则将均值和标准差移动到CPU上
            self.N.loc = self.N.loc.cpu()
            self.N.scale = self.N.scale.cpu()

        # self.N.loc = self.N.loc.cuda()
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.linear3(x)
        sigma = torch.exp(self.linear4(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
class Decoder_Anomalous(nn.Module):
    def __init__(self,latent_dims,embed_size):
        super(Decoder_Anomalous,self).__init__()
        self.linear1 = nn.Linear(latent_dims,int(embed_size/2))
        self.linear2 = nn.Linear(int(embed_size/2),embed_size)
        self.linear3 = nn.Linear(embed_size,embed_size)

    def forward(self,z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = self.linear3(z)
        return z

class VariationalEncoder_Property(nn.Module):
    def __init__(self, latent_dims, embed_size):
        super(VariationalEncoder_Property, self).__init__()
        self.linear1 = nn.Linear(embed_size, int(embed_size / 2))
        self.linear2 = nn.Linear(int(embed_size / 2), int(latent_dims/2))
        # ####两层，到16停止
        # self.linear3 = nn.Linear(int(latent_dims/2), int(latent_dims/2))
        # self.linear4 = nn.Linear(int(latent_dims/2), int(latent_dims/2))
        ###三层，到8停止
        self.linear5 = nn.Linear(int(latent_dims / 2), int(latent_dims / 4))
        self.linear3 = nn.Linear(int(latent_dims/4), int(latent_dims/4))
        self.linear4 = nn.Linear(int(latent_dims/4), int(latent_dims/4))
        ###四层，到4停止
        # self.linear5 = nn.Linear(int(latent_dims / 2), int(latent_dims / 4))
        # self.linear6 = nn.Linear(int(latent_dims/4),int(latent_dims/8))
        # self.linear3 = nn.Linear(int(latent_dims / 8), int(latent_dims / 8))
        # self.linear4 = nn.Linear(int(latent_dims / 8), int(latent_dims / 8))

        ###三层，到128维的到16
        # self.linear1 = nn.Linear(embed_size, int(embed_size / 2))
        # self.linear2 = nn.Linear(int(embed_size / 2), latent_dims)
        # self.linear5 = nn.Linear(latent_dims, int(latent_dims / 2))
        # self.linear3 = nn.Linear(int(latent_dims / 2), int(latent_dims / 2))
        # self.linear4 = nn.Linear(int(latent_dims / 2), int(latent_dims / 2))
        ###四层，128维的到8
        # self.linear1 = nn.Linear(embed_size, int(embed_size / 2))
        # self.linear2 = nn.Linear(int(embed_size / 2), latent_dims)
        # self.linear5 = nn.Linear(latent_dims, int(latent_dims / 2))
        # self.linear6 = nn.Linear(int(latent_dims / 2), int(latent_dims / 4))
        # self.linear3 = nn.Linear(int(latent_dims / 4), int(latent_dims / 4))
        # self.linear4 = nn.Linear(int(latent_dims / 4), int(latent_dims / 4))
        self.N = torch.distributions.Normal(0, 1)
        print("struct编码器维度:"+str(embed_size)+"中间层维度"+str(int(latent_dims/2)))
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            # 如果CUDA可用，则将均值和标准差移动到CUDA设备上
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
        else:
            # 如果CUDA不可用，则将均值和标准差移动到CPU上
            self.N.loc = self.N.loc.cpu()
            self.N.scale = self.N.scale.cpu()

        # self.N.loc = self.N.loc.cuda()
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        ###到8停止，三层
        x = F.relu(self.linear5(x))
        ###四层，到4停止
        # x = F.relu(self.linear5(x))
        # x = F.relu(self.linear6(x))
        # mu = self.linear3(x)
        # sigma = torch.exp(self.linear4(x))
        ###三层，128维到16
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear5(x))
        #
        # mu = self.linear3(x)
        # sigma = torch.exp(self.linear4(x))
        ###四层，128维到8
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear5(x))
        # x = F.relu(self.linear6(x))
        #
        mu = self.linear3(x)
        sigma = torch.exp(self.linear4(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z
class Decoder_Property(nn.Module):
    def __init__(self, latent_dims, embed_size):
        super(Decoder_Property, self).__init__()



        ###两层，到16停止
        # self.linear1 = nn.Linear(int(latent_dims / 2), latent_dims)
        # self.linear2 = nn.Linear(latent_dims, embed_size)
        # self.linear3 = nn.Linear(embed_size, embed_size)
        ###三层，到8停止
        self.linear1 = nn.Linear(int(latent_dims / 4), int(latent_dims/2))
        self.linear2 = nn.Linear(int(latent_dims/2), latent_dims)
        self.linear3 = nn.Linear(latent_dims,embed_size)
        self.linear4 = nn.Linear(embed_size,embed_size)
        ###四层，到4停止
        # self.linear1 = nn.Linear(int(latent_dims / 8), int(latent_dims / 4))
        # self.linear2 = nn.Linear(int(latent_dims / 4), int(latent_dims/2))
        # self.linear3 = nn.Linear(int(latent_dims / 2), latent_dims)
        # self.linear4 = nn.Linear(latent_dims, embed_size)
        # self.linear5 = nn.Linear(embed_size, embed_size)
        ###三层，128维到16
        # self.linear1 = nn.Linear(int(latent_dims / 2), latent_dims)
        # self.linear2 = nn.Linear(latent_dims, latent_dims * 2)
        # self.linear3 = nn.Linear(int(embed_size / 2), embed_size)
        # self.linear4 = nn.Linear(embed_size, embed_size)
        ###四层，128维到8维
        # self.linear1 = nn.Linear(int(latent_dims / 4), int(latent_dims / 2))
        # self.linear2 = nn.Linear(int(latent_dims / 2), latent_dims)
        # self.linear3 = nn.Linear(latent_dims, latent_dims * 2)
        # self.linear4 = nn.Linear(int(embed_size / 2), embed_size)
        # self.linear5 = nn.Linear(embed_size, embed_size)
    def forward(self, z):
        ##两层，到16停止
        # z = F.relu(self.linear1(z))
        # z = F.relu(self.linear2(z))
        # z = self.linear3(z)
        ###三层，到8停止
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = self.linear4(z)
        ###四层，到4停止
        # z = F.relu(self.linear1(z))
        # z = F.relu(self.linear2(z))
        # z = F.relu(self.linear3(z))
        # z = F.relu(self.linear4(z))
        # z = self.linear5(z)
        ###三层，128维到16维
        # z = F.relu(self.linear1(z))
        # z = F.relu(self.linear2(z))
        # z = F.relu(self.linear3(z))
        # z = self.linear4(z)
        ###四层，128维到8维
        # z = F.relu(self.linear1(z))
        # z = F.relu(self.linear2(z))
        # z = F.relu(self.linear3(z))
        # z = F.relu(self.linear4(z))
        # z = self.linear5(z)

        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_struct_dims,latent_anomalous_dim,latent_property_dim,struct_size,anomalous_size,property_size):
        super(VariationalAutoencoder,self).__init__()
        ##结构编码器
        self.encoder_struct = VariationalEncoder_Struct(latent_struct_dims,struct_size)
        self.decoder_struct = Decoder_Struct(latent_struct_dims,struct_size)
        ###恶意编码器
        self.encoder_anomalous = VariationalEncoder_Anomalous(latent_anomalous_dim, anomalous_size)
        self.decoder_anomalous = Decoder_Anomalous(latent_anomalous_dim, anomalous_size)

        ###属性编码器
        self.encoder_property = VariationalEncoder_Property(latent_property_dim, property_size)
        self.decoder_property = Decoder_Property(latent_property_dim, property_size)
    def forward(self,x,feature_type):
        if feature_type == "struct":
            z = self.encoder_struct(x)
            return self.decoder_struct(z)
        if feature_type == "anomalous":
            z = self.encoder_anomalous(x)
            return self.decoder_anomalous(z)
        if feature_type == "property":
            z = self.encoder_property(x)
            return self.decoder_property(z)

