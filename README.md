# yolov3-network-slimming

[![LICENSE](https://img.shields.io/badge/license-NPL%20(The%20996%20Prohibited%20License)-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

将[Learning Efficient Convolutional Networks Through Network Slimming (ICCV 2017)](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)应用在yolov3和yolov2上<bar>

# 环境
pytorch 0.41 

window 10

# 如何使用
1.对原始weights文件进行稀疏化训练

python sparsity_train.py -sr --s 0.0001 --image_folder coco.data  --cfg yolov3.cfg --weights yolov3.weights 

2.剪枝

python prune.py --cfg yolov3.cfg --weights checkpoints/yolov3_sparsity_100.weights --percent 0.3

3.对剪枝后的weights进行微调<bar>
  
python sparsity_train.py --image_folder coco.data  --cfg prune_yolov3.cfg --weights prune_yolov3.weights 

# 关于new_prune.py
new_prune更新了算法，现在可以确保不会有某一层被减为0的情况发生，参考[RETHINKING THE SMALLER-NORM-LESSINFORMATIVE ASSUMPTION IN CHANNEL PRUNING OF CONVOLUTION LAYERS(ICLR 2018)](https://arxiv.org/abs/1802.00124?context=cs)对剪枝后bn层β系数进行了保留

# 待完成
coco测试
