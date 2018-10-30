from __future__ import division
from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import defaultdict


class shortcutLayer(nn.Module):
    def __init__(self,froms):
        super(shortcutLayer, self).__init__()
        self.froms = froms


class Route(nn.Module):
    def __init__(self,layers):
        super(Route, self).__init__()
        self.layers =layers


class DetectionLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim,ignore_thresh ):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = ignore_thresh
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(reduction='elementwise_mean')  # Coordinate loss 均方损失函数
        self.bce_loss = nn.BCELoss(reduction='elementwise_mean')  # Confidence loss 适用于多目标分类
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, targets=None):
        nA = self.num_anchors  #x.size=1*255*13*13
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG #    stride=416/13=32

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
                            #(B*3*85*13*13)------>(B*3*13*13*85)
        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs (1*3*13*13)
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid          (1*1*13*13) [0...12],[0...12]
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        # [0...0],[1....1]....[12...12]
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        #anchors在特征图的形状
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        # anchors   1*3*1*1
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors  (B*3*13*13*4)
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        #B*3*13*13 +1*1*13*13,特征图上的中心点
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        #B*3*13*13 * 1*3*1*1，,特征图上的长宽大小
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:
            # target是一个B*50*5的矩阵
            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals)

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)
            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask
            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])  # x是中心点相对的偏移
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + \
                        self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (   #(1*(3*13*13)*4)
                    pred_boxes.view(nB, -1, 4) * stride,
                    #1*(3*13*13)*1
                    pred_conf.view(nB, -1, 1),
                    #1*(3*13*13)*80
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            #print(output.shape)
            return output


def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    block ={}
    blocks =[]
    for line in lines:
        if line[0] == '[':
            if len(block)!= 0:
                blocks.append(block)
                block ={}
            block['type'] = line[1:-1].rstrip()
        else:
            key,value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        # check the type of block
        # create a new module for the block
        # append to module_list
        if (x["type"] == "convolutional"):
        # Get the info about the layer
            activation = x["activation"]
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            if batch_normalize:
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
                module.add_module("conv_with_bn_{0}".format(index), conv)
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            else:
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
                module.add_module("conv_without_bn_{0}".format(index), conv)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

            # If it's an upsampling layer
            # We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)
            # If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            # Start of a route
            start = int(x["layers"][0])
            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = Route([start,end])
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

            # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            froms = int(x['from'])
            shortcut = shortcutLayer(froms)
            module.add_module("shortcut_{}".format(index), shortcut)
            # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            num_classes = int(x["classes"])
            img_height = int(net_info["height"])
            ignore_thresh = float(x["ignore_thresh"])
            detection = DetectionLayer(anchors,num_classes,img_height,ignore_thresh)
            module.add_module("Detection_{}".format(index), detection)


        elif x["type"] == "maxpool":
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            pool = nn.MaxPool2d(stride=stride,kernel_size=kernel_size)
            module.add_module("maxpool_{0}".format(index), pool)
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.img_size = self.net_info["height"]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    def forward(self, x, targets=None):
        is_training = targets is not None
        modules = self.blocks[1:]
        outputs = [] #We cache the outputs for the route layer
        layer_outputs = []
        self.losses = defaultdict(float)
        write = 0  # This is explained a bit later
        for i, module in enumerate(modules):
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut": # 残差
                from_ = int(module["from"])
                x = outputs[i + from_]+outputs[i - 1]
            elif module_type =="maxpool":
                x = self.module_list[i](x)
            elif module_type == "yolo":
                if is_training:
                    x, *losses = self.module_list[i](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = self.module_list[i](x)
                layer_outputs.append(x)
            outputs.append(x)

        self.losses["recall"] /= 3
        self.losses["precision"] /= 3
        return sum(layer_outputs) if is_training else torch.cat(layer_outputs, 1)


    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")
        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        # Needed to write header when saving weights
        self.header_info = header
        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            # If module_type is convolutional load weights
            # Otherwise ignore.
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]

                if (batch_normalize):
                    bn = model[1]
                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()
                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
        print("done!")


    def save_weights(self,path,cutoff=-1):
        """save layers between 0 and cutoff (cutoff = -1 -> all are saved)"""
        fp = open(path,'wb')
        self.header_info[3]=self.seen
        self.header_info.tofile(fp)
        for i in range(len(self.module_list[:cutoff])):
            module_type = self.blocks[i + 1]["type"]
            # If module_type is convolutional load weights
            # Otherwise ignore.
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]
                if (batch_normalize):
                    bn = model[1]
                    bn.bias.data.cpu().numpy().tofile(fp)
                    bn.weight.data.cpu().numpy().tofile(fp)
                    bn.running_mean.data.cpu().numpy().tofile(fp)
                    bn.running_var.data.cpu().numpy().tofile(fp)
                else:
                    conv.bias.data.cpu().numpy().tofile(fp)
                conv.weight.data.cpu().numpy().tofile(fp)
        fp.close()
