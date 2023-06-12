import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import random
from data import SEIDataset
from model.AlexNet import AlexNet
from model.VGG_16_1D import VGG_16_1D
from model.DAConv_VGG_16_1D import DAConv_VGG_16_1D
from model.ResNet_50_1D import ResNet_50_1D
from model.DAConv_ResNet_50_1D import DAConv_ResNet_50_1D, DAConv_ResNet_34_1D
from model.CVCNN import CVCNN
from model.AFFNet import AFFNet
from utils import count_parameters, lr_schedule_func_builder, Logger, show_loss_acc_curve
from config import cfg
import pickle
import time
from utils import get_mean_std
from data import SEIDataset
import time
from torch_geometric.nn import TopKPooling,SAGEConv,GCNConv,SAGPooling
import numpy as np
from torch_geometric.data import Data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = {'AlexNet': AlexNet,
         'VGG': VGG_16_1D,
         'VGG-DAConv': DAConv_VGG_16_1D,
         'ResNet': ResNet_50_1D,
         'ResNet50-DAConv': DAConv_ResNet_50_1D,
         'ResNet34-DAConv': DAConv_ResNet_34_1D,
         'CVCNN': CVCNN,
         'AFFNet': AFFNet} 
with open('{}train_mean_std.pkl'.format(cfg['train_data_dir']), 'rb') as f:
    mean_std = pickle.load(f)
mean, std = mean_std[0], mean_std[1]
dataset1 = SEIDataset(data_file=cfg['h5_file'], split='test') 
mean1, std1 = get_mean_std(dataset1, ratio=1)
def normalize(x, m, s):
    batch_size = x.size(0) 
    length = x.size(2)
    m = torch.from_numpy(m)
    m = m.unsqueeze(0).unsqueeze(2) 
    m = m.repeat(batch_size, 1, length) 
    s = torch.from_numpy(s) 
    s = s.unsqueeze(0).unsqueeze(2) 
    x = (x-m)/s 
    return x
def bce_with_logits(logits, labels):
    assert logits.dim() == 2 
    loss = F.binary_cross_entropy_with_logits(logits, labels) 
    return loss
def compute_score_with_logits(logits, labels): 
    with torch.no_grad():
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*labels.size()) 
        one_hots.scatter_(1, logits.view(-1, 1), 1) 
        scores = (one_hots * labels) 
        return scores
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)
    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) 
        u = u / self.scale 
        if mask is not None:
            u = u.masked_fill(mask, -np.inf)
        attn = self.softmax(u) 
        output = torch.bmm(attn, v) 
        return output
class FSL(nn.Module):
    def __init__(self, sample_len, in_channels, out_channels, kernel_size, stride, padding, spatial_att_enable=True, channel_att_enable=True, residual_enable=True):
        super(FSL, self).__init__()
        self.out_channels=out_channels
        self.sample_len = sample_len #长度
        self.out_len = int((sample_len+2*padding-(kernel_size-1)-1)/stride +1) 
        self.spatial_att_enable = spatial_att_enable
        self.channel_att_enable = channel_att_enable 
        self.residual_enable = residual_enable 
        self.conv_branch = nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding) 
        self.multihead_attn = nn.MultiheadAttention(self.out_len, 1,batch_first='True')
        self.attention = ScaledDotProductAttention(scale=np.power(out_channels, 0.5))
        if spatial_att_enable:
            self.spatial_att_conv = nn.Conv1d(in_channels, 1, kernel_size, 2)
            self.spatial_att_pool = nn.MaxPool1d(kernel_size, 2) 
            self.spatial_att_deconv = nn.ConvTranspose1d(1, 1, kernel_size, 2) 
            self.upsample = nn.Upsample(size=(self.out_len), mode='linear') 
        if channel_att_enable:
            self.channel_att_fc1 = nn.Linear(sample_len, out_channels)
            self.channel_att_fc2 = nn.Linear(in_channels, 1) 
    def forward(self, x):
        conv_out = self.conv_branch(x)
        if self.spatial_att_enable: 
            satt_map = F.relu(self.spatial_att_conv(x)) 
            satt_map = self.spatial_att_pool(satt_map) 
            satt_map = F.relu(self.spatial_att_deconv(satt_map)) 
            satt_map = self.upsample(satt_map) 
            satt_map = torch.sigmoid(satt_map)
            if self.residual_enable: 
                conv_out2=torch.repeat_interleave(satt_map,self.out_channels,dim=1)
            else:
                conv_out = conv_out * satt_map.unsqueeze(1) 
        if self.channel_att_enable:
            catt_map = torch.tanh(self.channel_att_fc1(x))
            catt_map = torch.transpose(catt_map, 1, 2) 
            catt_map = self.channel_att_fc2(catt_map) 
            catt_map = torch.sigmoid(catt_map) 
            catt_map = torch.sum(catt_map, dim=-1, keepdim=True) 
            if self.residual_enable: 
                conv_out1=torch.repeat_interleave(catt_map,self.out_len,dim=2)
            else:
                conv_out = conv_out * catt_map 
        x=self.attention(conv_out2,conv_out1,conv_out)
        x=conv_out+x
        return x 
class AlexNet(nn.Module):
    '''简化版的alexnet,效果更好'''
    def __init__(self, cfg):
        super(AlexNet, self).__init__()
        self.samp_len = cfg['sample_len']
        self.conv1 = FSL(self.samp_len, 2, 64, 11, 4, 0) 
        self.samp_len = int((self.samp_len - 11) / 4 + 1) 
        self.pool = nn.MaxPool1d(3, 2)
        self.samp_len = int((self.samp_len - 3) / 2 + 1)
        self.conv2 = FSL(self.samp_len, 64, 128, 5, 1, 2) 
        self.samp_len = int((self.samp_len - 5 + 4) / 1 + 1) 
        self.samp_len = int((self.samp_len - 3) / 2 + 1)
        self.conv3 = FSL(self.samp_len, 128, 128, 3, 1, 1)
        self.samp_len = int((self.samp_len - 1) / 1 + 1)
        self.conv4 = FSL(self.samp_len, 128, 64, 3, 1, 1) 
        self.samp_lenI = cfg['sample_len']
        self.conv1I = FSL(self.samp_lenI, 1, 64, 11, 4, 0)
        self.samp_lenI = int((self.samp_lenI - 11) / 4 + 1) 
        self.poolI = nn.MaxPool1d(3, 2) 
        self.samp_lenI = int((self.samp_lenI - 3) / 2 + 1) 
        self.conv2I = FSL(self.samp_lenI, 64, 128, 5, 1, 2) 
        self.samp_lenI = int((self.samp_lenI - 5 + 4) / 1 + 1) 
        self.samp_lenI = int((self.samp_lenI - 3) / 2 + 1)
        self.conv3I = FSL(self.samp_lenI, 128, 128, 3, 1, 1) 
        self.samp_lenI = int((self.samp_lenI - 1) / 1 + 1) 
        self.conv4I = FSL(self.samp_lenI, 128, 64, 3, 1, 1) 
        self.samp_lenQ = cfg['sample_len']
        self.conv1Q = FSL(self.samp_lenQ, 1, 64, 11, 4, 0)
        self.samp_lenQ = int((self.samp_lenQ - 11) / 4 + 1)
        self.poolQ = nn.MaxPool1d(3, 2)
        self.samp_lenQ = int((self.samp_lenQ - 3) / 2 + 1)
        self.conv2Q = FSL(self.samp_lenQ, 64, 128, 5, 1, 2)
        self.samp_lenQ = int((self.samp_lenQ - 5 + 4) / 1 + 1)
        self.samp_lenQ = int((self.samp_lenQ - 3) / 2 + 1)
        self.conv3Q = FSL(self.samp_lenQ, 128, 128, 3, 1, 1) 
        self.samp_lenQ = int((self.samp_lenQ - 1) / 1 + 1) 
        self.conv4Q = FSL(self.samp_lenQ, 128, 64, 3, 1, 1) 
        self.attention = ScaledDotProductAttention(scale=np.power(14, 0.5))
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(896, 128) 
        self.fc2 = nn.Linear(128, cfg['n_classes'])
    def forward(self, x):
        y=x.flatten(1)
        I=y[:,:512]
        Q=y[:,512:]
        I=I.unsqueeze(1)
        Q=Q.unsqueeze(1)
        x = F.relu(self.conv1(x))    
        x = self.pool(x)             
        x = F.relu(self.conv2(x))    
        x = self.pool(x)              
        x = F.relu(self.conv3(x))    
        x = F.relu(self.conv4(x))    
        x = self.pool(x)              
        I = F.relu(self.conv1I(I))     
        I = self.poolI(I)              
        I = F.relu(self.conv2I(I))     
        I = self.poolI(I)              
        I = F.relu(self.conv3I(I))     
        I = F.relu(self.conv4I(I))     
        I = self.poolI(I)               
        Q = F.relu(self.conv1Q(Q))     
        Q = self.poolQ(Q)             
        Q = F.relu(self.conv2Q(Q))    
        Q = self.poolQ(Q)              
        Q = F.relu(self.conv3Q(Q))    
        Q = F.relu(self.conv4Q(Q))    
        Q = self.poolQ(Q)              
        attn_output= self.attention(I,Q,x)
        x = attn_output + x
        x = x.view(-1, self.num_flat_features(x)) 
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    def num_flat_features(self, x): 
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
def train(epoch): 
    dataset = iter(train_loader) 
    pbar = tqdm(dataset)
    epoch_score = 0.0  
    epoch_loss = 0.0   
    moving_score = 0   
    moving_loss = 0    
    n_samples = 0
    net.train(True) 
    start_time = time.time() 
    for IQ_slice, label in pbar: 
        IQ_slice = normalize(IQ_slice, mean, std) 
        
        IQ_slice, label = ( 
            IQ_slice.to(device),
            label.to(device),
        )
        n_samples += label.size(0)
        net.zero_grad() 
        output = net(IQ_slice) 
        target = torch.zeros_like(output).scatter_(1, label.view(-1, 1), 1) 
        loss = criterion(output, target) 
        loss.backward() 
        clip_grad_norm_(net.parameters(), 0.25) 
        optimizer.step() 
        batch_score = compute_score_with_logits(output, target).sum() 
        epoch_loss += float(loss.data.item()) * target.size(0)
        epoch_score += float(batch_score) 
        moving_loss = epoch_loss / n_samples
        moving_score = epoch_score / n_samples 
        loss_acc['train_loss'].append(float(loss.data.item()))
        loss_acc['train_acc'].append(float(batch_score)/label.size(0)) 
        loss_acc['train_moving_loss'].append(moving_loss)
        loss_acc['train_moving_acc'].append(moving_score)
        pbar.set_description(
            'Train Epoch: {}; Loss: {:.6f}; Acc: {:.6f}'.format(epoch + 1, moving_loss, moving_score))
    end_time = time.time()
    print(end_time-start_time) 
    logger.write('Epoch: {:2d}: Train Loss: {:.6f}; Train Acc: {:.4f}'.format(epoch+1, moving_loss, moving_score)) 
def test(epoch): 
    dataset = iter(test_loader) 
    pbar = tqdm(dataset) 
    epoch_score = 0.0 
    epoch_loss = 0.0 
    moving_score = 0
    moving_loss = 0
    n_samples = 0 
    net.eval() 
    with torch.no_grad(): 
        for IQ_slice, label in pbar:
            IQ_slice = normalize(IQ_slice, mean1, std1) 
            IQ_slice, label = (
                IQ_slice.to(device),
                label.to(device),
            )
            n_samples += label.size(0)
            output = net(IQ_slice)
            target = torch.zeros_like(output).scatter_(1, label.view(-1, 1), 1)
            loss = criterion(output, target)
            batch_score = compute_score_with_logits(output, target).sum()
            epoch_loss += float(loss.data.item()) * target.size(0)
            epoch_score += float(batch_score)
            moving_loss = epoch_loss / n_samples
            moving_score = epoch_score / n_samples
            loss_acc['test_loss'].append(float(loss.data.item()))
            loss_acc['test_acc'].append(float(batch_score)/label.size(0))
            loss_acc['test_moving_loss'].append(moving_loss)
            loss_acc['test_moving_acc'].append(moving_score)

            pbar.set_description(
                'Val Epoch: {}; Loss: {:.6f}; Acc: {:.6f}'.format(epoch + 1, moving_loss, moving_score))
    logger.write('Val: {:2d}: Loss: {:.6f}; Acc: {:.4f}'.format(epoch+1, moving_loss, moving_score))
if __name__ == '__main__':
    np.random.seed(101) 
    random.seed(101) 
    torch.manual_seed(101) 
    print('Loading data...')
    train_dataset = SEIDataset(data_file=cfg['h5_file'], split='train')
    test_dataset = SEIDataset(data_file=cfg['h5_file'], split='test')
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=True) 
    print('Creating Model...')
    net =AlexNet(cfg).to(device)
    n_params = count_parameters(net) 
    print("model: {:,}  parameters".format(n_params)) 
    ssl_checkpoint = '{}/ssl_checkpoint_{}.pth'.format(cfg['checkpoint_path'] + cfg['model'], str(cfg['ssl_n_epoch']).zfill(2)) 
    criterion = bce_with_logits 
    optimizer = optim.Adam(net.parameters(), lr=cfg['lr']) 
    sched = LambdaLR(optimizer, lr_lambda=lr_schedule_func_builder()) 
    checkpoint_path = cfg['checkpoint_path'] + cfg['model'] 
    if os.path.exists(checkpoint_path) is False: 
        os.mkdir(checkpoint_path)
    logger = Logger(os.path.join(checkpoint_path, "log.txt")) 
    for k, v in cfg.items(): 
        logger.write(k+': {}'.format(v))
    loss_acc = {
        'train_loss': [],
        'train_acc': [],
        'train_moving_loss': [],
        'train_moving_acc': [],
        'test_loss': [],
        'test_acc': [],
        'test_moving_loss': [],
        'test_moving_acc': []} 
    print('Starting train...')
    for epoch in range(cfg['n_epoch']):
        print('\n lr={}'.format(optimizer.state_dict()["param_groups"][0]["lr"])) 
        train(epoch) 
        test(epoch) 
        sched.step() 
        if (epoch+1) % 5 == 0:
            with open('{}/checkpoint_{}.pth'.format(checkpoint_path, str(epoch + 1).zfill(2)), 'wb') as f: 
                torch.save(net.state_dict(), f)
            with open('{}/optim_checkpoint_{}.pth'.format(checkpoint_path, str(epoch + 1).zfill(2)), 'wb') as f: 
                torch.save(optimizer.state_dict(), f)
    with open('{}/loss_acc.pkl'.format(checkpoint_path), 'wb') as f: 
        pickle.dump(loss_acc, f)
    show_loss_acc_curve('{}/loss_acc.pkl'.format(checkpoint_path), checkpoint_path)





























