from __future__ import division

from yolomodel import *
from util  import *
from parse_config import *
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler


def arg_parse():
    parser = argparse.ArgumentParser(description="YOLO v3 Train")
    parser.add_argument("--image_folder", type=str, default=r"D:\yolotest\data\coco.data", help="path to dataset")
    parser.add_argument("--epochs",dest="epochs",help="epochs",default=16)
    parser.add_argument("--cfg",dest="cfgfile",help="网络模型",
                        default=r"D:/yolotest/cfg/yolov3.cfg",type=str)
    parser.add_argument("--weights",dest="weightsfile",help="权重文件",
                        default=r"D:/yolotest/cfg/yolov3.weights",type=str)
    parser.add_argument("--reso", dest='reso', help="resize图片大小",
                        default="416", type=str)
    parser.add_argument("--n_cpu",dest='n_cpu',type=int,default=2,help="torch多线程核数")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
    parser.add_argument("-sr", dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.0001,
                        help='稀疏化比率')
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
    )
    return parser.parse_args()

# 只稀疏化非shortcut的层
def updateBN(model,s,donntprune):
    for k,m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            if k not in donntprune:
                m.weight.grad.data.add_(s*torch.sign(m.weight.data))


def train():
    args = arg_parse()
    cuda = torch.cuda.is_available() and args.use_cuda
    data_config = parse_data_config(args.image_folder)
    train_path = data_config["train"]
    classes_path = data_config["names"]
    classes = load_classes(classes_path)
    num_classes = len(classes)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initiate model
    print("load network")
    model = Darknet(args.cfgfile)
    print("done!")
    print("load weightsfile")
    model.load_weights(args.weightsfile)
    # Get hyper parameters
    hyperparams = model.blocks[0]
    learning_rate = float(hyperparams["learning_rate"])
    momentum = float(hyperparams["momentum"])
    decay = float(hyperparams["decay"])
    burn_in = int(hyperparams["burn_in"])
    inp_dim = int(model.net_info["height"])
    batch_size = int(hyperparams["batch"])
    if cuda:
        model = model.cuda()
    model.train()
    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        ListDataset(train_path,img_size=inp_dim), batch_size=batch_size, shuffle=False, num_workers=args.n_cpu
    )
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum,weight_decay=decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    #记录哪些是shortcut层
    donntprune = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, shortcutLayer):
            x = k + m.froms - 8
            donntprune.append(x)
            x = k - 3
            donntprune.append(x)
    # print(donntprune)

    for epoch in range(args.epochs):
        exp_lr_scheduler.step(epoch)
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)
            optimizer.zero_grad()
            loss = model(imgs, targets)
            loss.backward()
            if args.sr:
                updateBN(model,args.s,donntprune)
            optimizer.step()
            print(
                "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    epoch,
                    args.epochs,
                    batch_i,
                    len(dataloader),
                    model.losses["x"],
                    model.losses["y"],
                    model.losses["w"],
                    model.losses["h"],
                    model.losses["conf"],
                    model.losses["cls"],
                    loss.item(),
                    model.losses["recall"],
                    model.losses["precision"],
                )
            )
            model.seen += imgs.size(0)

        if epoch % args.checkpoint_interval == 0:
            if args.sr:
                model.train(False)
                total = 0
                for k, m in enumerate(model.modules()):
                    if isinstance(m, nn.BatchNorm2d):
                        if k not in donntprune:
                            total += m.weight.data.shape[0]
                bn = torch.zeros(total)
                index = 0
                for k, m in enumerate(model.modules()):
                    if isinstance(m, nn.BatchNorm2d):
                        if k not in donntprune:
                            size = m.weight.data.shape[0]
                            bn[index:(index + size)] = m.weight.data.abs().clone()
                            index += size
                y, i = torch.sort(bn)  # y,i是从小到大排列所有的bn，y是weight，i是序号
                number = int(len(y)/5)  # 将总类分为5组
                # 输出稀疏化水平
                print("0~20%%:%d,20~40%%:%d,40~60%%:%d,60~80%%:%d,80~100%%:%d"%(y[number],y[2*number],y[3*number],y[4*number],y[-1]))
                
            model.save_weights("%s/yolov3_sparsity_%d.weights" % (args.checkpoint_dir, epoch))
            print("save weights in %s/yolov3_sparsity_%d.weights" % (args.checkpoint_dir, epoch))
            model.train()


if __name__ =='__main__':
    train()
