# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:08:33 2018

@author: tm
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:39:04 2018

@author: tm
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:51:29 2018

@author: tm
"""

from osgeo import gdal 
#import gdal
import cv2

from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session


import numpy as np
import tensorflow as tf
from ssd import SSD300
from ssd_utils import BBoxUtility

import saveto_shapefiles as save_polygon

#%matplotlib inline
#plt.rcParams['figure.figsize'] = (8, 8)
#plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))


#voc_classes = ['baseball','ground_track_field']

def predict_to_coord(model_weights, image_path, band=[4,3,2], conf_threshold=0.5):
    '''
    Input :
        model_weights 训练好的模型权重
        image_path  要预测的图像 eg: r'...\to_xiangyu\geo_.tif'
        band    选区的波段 第一波段标号为 1
        cconf_threshold     输出结果的阈值大小
        
        
    Output :
        Prj_id   影像的 地理参考系编号
        result_items  list （行号，列号，置信度，坐标
                            行号，列号，置信度，坐标）
                            ...
    
    
    '''
    voc_classes = ['popp']
    NUM_CLASSES = len(voc_classes) + 1
    
    input_shape=(300, 300, 3)
    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    
    #model.load_weights('./checkpoints_330/SSD_4_100_0.0002.hdf5', by_name=True)
    model.load_weights(model_weights, by_name=True)
    bbox_util = BBoxUtility(NUM_CLASSES)
    
    
    #dataset = gdal.Open(r'C:\Liuxiangyu\毕业实验\基础数据\to_xiangyu\geo_.tif')
    dataset =gdal.Open(image_path)
    
    GeoTransform =dataset.GetGeoTransform()
    x0, y0 = GeoTransform[0], GeoTransform[3]           #影像左上方坐标
    x_pixel, y_pixel = GeoTransform[1], GeoTransform[5]  #行列 分辨率
    
    Prj_id =int(dataset.GetProjection().split('"')[-2])  #投影参数，eg: 4326
    
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_band = dataset.RasterCount
    #im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    
    #get shapelike(width, height, band)
    #im_data = np.swapaxes(im_data,0,1)
    #im_data = np.swapaxes(im_data,2,1) 
    
    if np.max(band) > im_band:
        
        raise Exception('the parameter  band is out of all bands')

#    if im_band == 5:
#        im_datas =im_data[...,band]
#        
#    elif im_band == 4:
#        im_datas =im_data[...,[2,1,0]]
#    else:
#        raise Exception('only can open rasters which has 4 or 5 bands')
    band =[b-1 for b in band]    
    #im_datas =im_data[...,band]
    
    
    #input_size =300
    y_num =int(im_height/overlap)
    x_num =int(im_width/overlap)
#    x_num =math.ceil(im_width/input_size)
#    y_num =math.ceil(im_height/input_size)
    result_items =[]
    for row in range(y_num):
        for column in range(x_num):
            
            y_st, x_st = overlap* row, overlap* column
            
            if im_width >= x_st + cut_size:
                if im_height >= y_st + cut_size:
                    im_data = dataset.ReadAsArray(x_st, y_st,  cut_size,  cut_size)
                else:
                    im_data = dataset.ReadAsArray(x_st, y_st,  cut_size, im_height - y_st)
            else:
                if im_height >= y_st + cut_size:
                    im_data = dataset.ReadAsArray(x_st, y_st, im_width - x_st,  cut_size)
                else:
                    im_data = dataset.ReadAsArray(x_st, y_st, im_width - x_st, im_height - y_st)
            im_data[im_data>65534] = 0
            
            x = np.swapaxes(im_data,0,1)
            x = np.swapaxes(x,2,1)
            im_data = x[..., band]
#            im_data = im_datas[row*input_size:(row+1)*input_size, column*input_size:(column+1)*input_size, :]
            im_data_max =im_data.max()
            im_data_min =im_data.min()
            inputs = []
            images = []
            mask =im_data.copy()
            mask[mask > 0] = 1
            data_radio =np.sum(mask)/(3*cut_size**2)  #计算无效区的像素占比， 少于一半的不参与预测
            if data_radio>0.4:
                
                input_data = (im_data-im_data_min)/(im_data_max-im_data_min)*255
                input_data =cv2.resize(input_data, (300,300))
                images.append(input_data)
                inputs.append(input_data)
            
                inputs = preprocess_input(np.array(inputs))
                preds = model.predict(inputs, batch_size=1, verbose=1)
                results = bbox_util.detection_out(preds)
    
    
                for i, img in enumerate(images):
                    # Parse the outputs.
                    #det_label =results[i][:, 0]
                    try:
                        
                        det_conf = results[i][:, 1]
                        det_xmin = results[i][:, 2]
                        det_ymin = results[i][:, 3]
                        det_xmax = results[i][:, 4]
                        det_ymax = results[i][:, 5]
                    
                        # Get detections with confidence higher than 0.6.
                        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_threshold]
                    
                        top_conf = det_conf[top_indices]
                        #top_label_indices = det_label[top_indices].tolist()
                        top_xmin = det_xmin[top_indices]
                        top_ymin = det_ymin[top_indices]
                        top_xmax = det_xmax[top_indices]
                        top_ymax = det_ymax[top_indices]
                    except:
                        print('this image has no result, has pass')
                        pass
                
                    for i in range(top_conf.shape[0]):
#                        xmin = int(round(top_xmin[i] * img.shape[1]))
#                        ymin = int(round(top_ymin[i] * img.shape[0]))
#                        xmax = int(round(top_xmax[i] * img.shape[1]))
#                        ymax = int(round(top_ymax[i] * img.shape[0]))
                        score = top_conf[i]
                        
                        rel_xmin =x0 + x_pixel*cut_size*(top_xmin[i]+ column)
                        rel_ymin =y0 + y_pixel*cut_size*(top_ymin[i]+ row)  #此处y_pixel为负数表示向下
                        rel_xmax =x0 + x_pixel*cut_size*(top_xmax[i]+ column)
                        rel_ymax =y0 + y_pixel*cut_size*(top_ymax[i]+ row)   #此处y_pixel为负数表示向下
                        
                        coods =((rel_xmin,rel_ymin),(rel_xmax,rel_ymin),(rel_xmax,rel_ymax),(rel_xmin,rel_ymax),(rel_xmin,rel_ymin))
                        item =(row,column,score,coods) #（行号，列号，置信度，坐标）
                        result_items.append(item)
    del dataset 
    
    return Prj_id, result_items

if __name__ == "__main__":
    
    
    image_tif = r'C:\Liuxiangyu\dataSet\zy3_3973\zy3_3973.tif'
    save_shp ="./predict_shp/poppy_zy3973_b432_zy3973_300_3189_4_300_0.0003.shp"
    
    cut_size =300   # 图像才切成的 大小
    overlap =300    # 裁切时 移动步长
    
    Prj_id, result_items = predict_to_coord('./checkpoints/zy3973_300_3189_4_300_0.0003.hdf5', 
                                            image_tif, 
                                            band=[4,3,2],
                                            conf_threshold=0.5)
    
    
    save_polygon.WriteVectorFile(save_shp, Prj_id, result_items)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    