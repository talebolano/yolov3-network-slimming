from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from util import *
import argparse
import os
from yolomodel import Darknet
from yolomodel import shortcutLayer


def arg_parse():
    parser = argparse.ArgumentParser(description="YOLO v3 Prune")
    parser.add_argument("--cfg",dest="cfgfile",help="网络模型",
                        default=r"D:/yolotest/cfg/yolov3.cfg",type=str)
    parser.add_argument("--weights",dest="weightsfile",help="权重文件",
                        default=r"D:/yolotest/cfg/yolov3.weights",type=str)
    parser.add_argument('--percent', type=float, default=0.3,help='剪枝的比例')
    return parser.parse_args()


args = arg_parse()
start = 0
CUDA = torch.cuda.is_available()
print("load network")
model = Darknet(args.cfgfile)
print("done!")
print("load weightsfile")
model.load_weights(args.weightsfile)
if CUDA:
    model.cuda()
# 根据shortcut找到不应该被裁的连接层，并记录其序号，放在donntprune中
donntprune = []
for k,m in enumerate(model.modules()):
    if isinstance(m, shortcutLayer):
        x= k+m.froms-8
        donntprune.append(x)
        x = k-3
        donntprune.append(x)
#print(donntprune)
#统计所有应该被裁的连接层的总大小
total = 0
for k,m in enumerate(model.modules()):
     if isinstance(m, nn.BatchNorm2d):
         if k not in donntprune:
            total += m.weight.data.shape[0]
#print(total)
bn = torch.zeros(total)
#print(bn)
index = 0
for k,m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        if k not in donntprune:
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
y, i = torch.sort(bn)# y,i是从小到大排列所有的bn，y是weight，i是序号
thre_index = int(total * args.percent)
thre = y[thre_index].cuda()
pruned = 0
cfg = []
cfg_mask = []
print('--'*30)
print("Pre-processing...")
for k, m in enumerate(model.modules()):
    #isinstance()函数来判断一个对象是否是一个已知的类型
    if isinstance(m, nn.BatchNorm2d):
        if k not in donntprune:
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()  # 掩模
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)# 直接修改m，直接改了model的值，并放在了model中
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                    format(k, mask.shape[0], int(torch.sum(mask))))
        else:
            dontp = m.weight.data.numel()
            mask = torch.ones(m.weight.data.shape)
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                    format(k, dontp, int(dontp)))
            cfg.append(int(dontp))
            cfg_mask.append(mask.clone())
pruned_ratio = pruned/total
print('Pre-processing Successful!')
print('--'*30)
#print(cfg)
# 写出被减枝的cfg文件
prunecfg = write_cfg(args.cfgfile,cfg)
newmodel = Darknet(prunecfg)
if CUDA:
    newmodel.cuda()
old_modules = list(model.modules())
new_modules = list(newmodel.modules())
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
print("pruning...")
v=0
for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    #print(m1)
    if isinstance(m0, nn.BatchNorm2d):# 向新模型中写入
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        #print(idx1.size)
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Sequential):
        for name in m0.named_children():
            if name[0].split("_")[0] == 'route':
                #print(old_modules[layer_id + 1].layers)
                #print(m0)
                ind = v+old_modules[layer_id + 1].layers[0]
                #print(ind)
                cfg_mask1 = cfg_mask[route_problem(model, ind)]
                #print(cfg_mask1.shape)
                if old_modules[layer_id + 1].layers[1]!=0:
                    ind =v + old_modules[layer_id + 1].layers[1]
                    #print(ind)
                    cfg_mask1 = cfg_mask1.unsqueeze(0)
                    #print(cfg_mask1.shape)
                    cfg_mask2 = cfg_mask[route_problem(model, ind)].unsqueeze(0).cuda()
                    #print(cfg_mask2.shape)
                    cfg_mask3 = torch.cat((cfg_mask1,cfg_mask2),1)
                    #print(cfg_mask3.shape)
                    cfg_mask1 = cfg_mask3.squeeze(0)
                    #print(cfg_mask1.shape)
                start_mask = cfg_mask1.clone()
                #print(cfg_mask1)
        # print(m0)
            elif "_".join(name[0].split("_")[0:-1]) == 'conv_with_bn':
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('Conv In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = old_modules[layer_id + 1].weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                new_modules[layer_id + 1].weight.data = w1.clone()
            elif "_".join(name[0].split("_")[0:-1]) == 'conv_without_bn':
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                w1 = old_modules[layer_id + 1].weight.data[:, idx0.tolist(), :, :].clone()
                new_modules[layer_id + 1].weight.data = w1.clone()
                new_modules[layer_id + 1].bias.data = old_modules[layer_id + 1].bias.data.clone()
                #print(new_modules[layer_id + 1].weight.data.size())
                print('Detect: In shape: {:d}, Out shape {:d}.'.format(new_modules[layer_id + 1].weight.data.size(1),
                      new_modules[layer_id + 1].weight.data.size(0)))
        v=v+1
    elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
print('--'*30)
print('prune done!')
print('pruned ratio %.3f'%pruned_ratio)
prunedweights = os.path.join('\\'.join(args.weightsfile.split("/")[0:-1]),"prune_"+args.weightsfile.split("/")[-1])
print('save weights file in %s'%prunedweights)
#保存新模型权重
newmodel.save_weights(prunedweights)
print('done!')
