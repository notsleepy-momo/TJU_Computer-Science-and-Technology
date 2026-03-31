from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom

from BagData import test_dataloader, train_dataloader
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet


def train(epo_num=50, show_vgg_params=False):

    vis = visdom.Visdom()    # 可视化工具

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 使用gpu或cpu，若安装cuda默认使用gpu
    """调用FCN.py中不同的vgg网络（自选一个）,进行定量实验分析"""
    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)   # backbone网络，用于特征提取
    """调用FCN.py中不同的FCN网络（自选一个）,进行定量实验分析"""
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)      # Architecture结构，用于分割
    fcn_model = fcn_model.to(device)    # 将模型加载至指定gpu
    """更改损失函数为交叉熵nn.CELoss(), 进行定量实验分析"""
    criterion = nn.BCELoss().to(device)
    """更改模型优化器，如：adam等（自选一个），并调整学习率lr和momentum参数,进行定量实验分析"""
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)
    # optimizer = optim.Adam(fcn_model.parameters(), lr=1e-2)
    
    all_train_iter_loss = [] # 训练loss存储list
    all_test_iter_loss = [] # 测试loss存储list

    # start timing
    prev_time = datetime.now()  # 训练起始时间记录
    for epo in range(epo_num):  # epo_num数据集训练次数
        
        train_loss = 0  #初始化loss
        fcn_model.train()  # 打开训练模式
        for index, (bag, bag_msk) in enumerate(train_dataloader):  # 加载数据，bag为数据样本，bag_msk为对应的二进制标签
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])

            bag = bag.to(device)  #数据加载至device上
            bag_msk = bag_msk.to(device)  #标签加载至device上

            optimizer.zero_grad()   #优化器梯度清空
            output = fcn_model(bag)  # 模型forward
            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, bag_msk)  # 计算模型预测与标签之间的损失
            loss.backward()  # 反向传播
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)  #loss添加至存储的list里
            train_loss += iter_loss  #训练集loss累加
            optimizer.step()  # 模型参数优化更新

            output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
            output_np = np.argmin(output_np, axis=1)
            bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160) 
            bag_msk_np = np.argmin(bag_msk_np, axis=1)

            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
                # vis.close()
                """结果可视化"""
                vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction')) 
                vis.images(bag_msk_np[:, None, :, :], win='train_label', opts=dict(title='label'))
                vis.line(all_train_iter_loss, win='train_iter_loss',opts=dict(title='train iter loss'))
        
        test_loss = 0
        fcn_model.eval()  # 模型打开测试模式
        with torch.no_grad():  # 测试阶段取消梯度
            for index, (bag, bag_msk) in enumerate(test_dataloader):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(bag)
                output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
                loss = criterion(output, bag_msk)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
                output_np = np.argmin(output_np, axis=1)
                bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160) 
                bag_msk_np = np.argmin(bag_msk_np, axis=1)
        
                if np.mod(index, 15) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result.')
                    # vis.close()
                    vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction')) 
                    vis.images(bag_msk_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                    vis.line(all_test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss'))

        """显示模型运行时间"""
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test loss = %f, %s'
                %(train_loss/len(train_dataloader), test_loss/len(test_dataloader), time_str))
        
        """每5个epoch保存一次模型到checkpoints路径"""
        if np.mod(epo, 5) == 0:
            torch.save(fcn_model, 'checkpoints/fcn_model_{}.pt'.format(epo))
            print('saveing checkpoints/fcn_model_{}.pt'.format(epo))


if __name__ == "__main__":

    train(epo_num=70, show_vgg_params=False)  #调用训练函数开始训练，训练epoch数为100
