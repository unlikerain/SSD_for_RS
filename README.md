"""
Created on Mon May 21 15:53:27 2018

@author: tm
"""

文件说明：
	SSD_train.py   模型训练文件
	make_trainDataset.py	用来制作数据集
	predict_on_pictures.py	对文件夹中的图片预测
	predict_to_shp.py	直接对整幅遥感图像预测，保存成shp
	ssd_network.py	SSD模型的网络结构
	ssd_lossFunction.py	模型的损失函数定义
	ssd_utils.py		输出框等工具
	prior_boxes_ssd300.pkl	训练时的初始化文件
