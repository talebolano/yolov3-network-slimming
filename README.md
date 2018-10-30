# yolov3-network-slimming
将[Learning Efficient Convolutional Networks Through Network Slimming (ICCV 2017)](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)应用在yolov3上<bar>

# 环境
pytorch 0.4 

window 10

# 如何使用
1.对原始weights文件进行稀疏化训练

python sparsity_train.py --s 0.0001 --image_folder coco.data  --cfg yolov3.cfg --weights yolov3.weights 

2.剪枝

python prune.py --cfg yolov3.cfg --weights checkpoints/yolov3_sparsity_100.weights --percent 0.3

3.对剪枝后的weights进行微调

