RGBD Anti-Spoofing：
1.增强样本：
 五官替换
 随机裁剪->Non
 加入遮挡->Photo/Pad
2.增大训练样本的区分性

=3.Focal_Loss
=4.SE-Block
=5.GAP->Stream_Module
=6.ShuffleNet 2.85M, FeatherNet 1.45M
=7.ShuffleNet 5.0M
8.多分支->特征拼接：
 RGB->SqueezeNet
 Depth->FeatherNet

9.->ncnn
10.DDWConv

11.bag of tricks
12.DWI 动态权重更新


DeMesh：
1.add color backgroud
