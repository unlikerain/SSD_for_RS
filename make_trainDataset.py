# -*- coding: utf-8 -*-
"""
Created on Wed May 16 16:30:02 2018

@author: tm
"""

"""
Created on Fri May 11 21:08:05 2018

程序用来快速方便的制作SSD模型的训练数据集
具体来说：
    对于普通的遥感图像，可以直接对整张图进行 目标的标注，需注意的是标注的须为 矩形
    标注的shp 应和遥感图像使用同一参考坐标系，否则可能会有BUG，但不一定报错
    
    只能针对单一目标进行标注，暂不支持对多目标的标注/
    
    在给定 要裁切的像元大小和 重叠大小后能够 灵活的根据波段组合裁出所需的 3波段 JPG文件
    同时生成 能直接用于SSD 模型的 .pkl文件
    

为了解决 对遥感数据进行 目标检测，制作数据集时需要将其裁切成jpg格式，
        然后在对每张进行手工标注，极花费时间，
        而且当需要改变数据集的尺寸时，又要重新标记，显然很愚。
        
        本程序便营运而生， 在一两个月之前，进行 手工标注的时候就想过这个问题
        只是当时觉得逻辑上很难把握，实现起来很是费劲，就没有进行下去
        
        现在的图像是512 *512 的输入SSD300网络时需要强制重采样为300* 300
        这种信息的丢失对遥感信息来说应该还较为敏感
        所以需要造几个工具来帮助进行图像的处理，
        
    历经一周的时间从构思到 到运行再到优化，终于完成。其中可能仍然有 BUG
    
    
    可优化的地方：：
                现在运行时内存占用极高，可以修改整合 read_img 函数，
                将每次读的图像大小设置为 cutsize
                
                已完成，
                
                2018.5.19 改进：
                    只保存参与训练的样本图片，使制作样本时间大大减少。
@author: tm
"""


from osgeo import gdal
from osgeo import ogr
import scipy
import numpy as np
import os,time
import sys
import pickle


    
def read_img(image_path, overlap):
    
    dataset =gdal.Open(image_path)
    
    GeoTransform =dataset.GetGeoTransform()
    x0, y0 = GeoTransform[0], GeoTransform[3]           #影像左上方坐标
    x_pixel, y_pixel = GeoTransform[1], GeoTransform[5]  #行列 分辨率
    
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    #im_band = dataset.RasterCount
    #im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    
    #处理背景值： 将65535 用0 替换
    #im_data[im_data>65534] = 0
    

    return dataset,im_width, im_height, x0, y0, x_pixel, y_pixel


def cut_to_jpg(dataset, row, colum, bands =[3,2,1], foder=r'D:\test\picccs', base_name='geo5644_'):
    
    if bands.__len__() !=3:
        raise Exception('bands 必须为三个波段，从1计数')
      
    y_st, x_st = overlap* row, overlap* colum
    #im_pic =im_pic[x_st: x_st+ cut_size, y_st: y_st+ cut_size]
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
    band_id =[b-1 for b in bands]
    im_pic =x[..., band_id] 
    #im_pic = srtetch_to_255(im_pic,throd_perscent =0.02)

    jpg_name =base_name +'{0:0>3}'.format(row)+'{0:0>3}'.format(colum) +'.jpg' #行和列号各占3位
    final_name =os.path.join(foder, jpg_name)
    scipy.misc.imsave(final_name, im_pic)
    #imageio.imwrite(final_name, im_pic)
    return jpg_name

def readShap(filename, x0, y0,x_pixel, y_pixel):  
    
    #将shp 的每个记录转化为 [x_min,xmax,ymin,y_max] 格式

    #为了支持中文路径，请添加下面这句代码   
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")     
    gdal.SetConfigOption("SHAPE_ENCODING","")    
    #注册所有的驱动   
    ogr.RegisterAll()    
    #数据格式的驱动  
    driver = ogr.GetDriverByName('ESRI Shapefile')  
    ds = driver.Open(filename)
    if ds is None:  
        raise ValueError ('Could not open shp'  ) 
    
    #geomlist = []
    geomlist_pix_cood =[]
    layer0 = ds.GetLayerByIndex(0)
    feature = layer0.GetNextFeature()    
    # 下面开始遍历图层中的要素    
    while feature is not None:    
        # 获取要素中的属性表内容    
        geom = feature.GetGeometryRef()
        
        poly_cood =geom.ExportToWkt()
         
        itme = poly_cood[10:-2]
        #以下四个 为地理坐标系下的坐标
        x_min =float(itme.split(',')[0].split(' ')[0])
        y_max =float(itme.split(',')[0].split(' ')[1])
        x_max =float(itme.split(',')[2].split(' ')[0])
        y_min =float(itme.split(',')[2].split(' ')[1])
        
        #cood = [x_min, x_max, y_min, y_max]  # 矩形的四角点坐标
        # 以下转化为 相对于整幅遥感影像的 像素坐标
        pixel_cood =[(x_min-x0)/x_pixel, (x_max-x0)/x_pixel, (y_max-y0)/y_pixel, (y_min-y0)/y_pixel]
        #geomlist.append(cood)  
        geomlist_pix_cood.append(pixel_cood)
        feature = layer0.GetNextFeature()

    return geomlist_pix_cood
 

def get_img_extent(i_row, i_column):
    
    # 得到第i_row, i_column处的图像占整幅影像的像素位置
    # 左上角为坐标系原点
    # overlap =256
    x_st, y_st = overlap* i_column, overlap* i_row

    
    rel_xmin = x_st
    rel_ymin = y_st
    rel_ymax =rel_ymin + cut_size  #此处y_pixel为负数表示向下
    rel_xmax =rel_xmin + cut_size

    return [rel_xmin,rel_xmax,rel_ymin,rel_ymax]


def is_overpass(cut_size,shp=[],img=[]):
    
    #函数用来判断两个矩形是否相交，并返回相交的矩形
    # 坐标系以 左上角为原点
    #输入格式：
    #shp =[x_min,xmax,ymin,y_max]
    #img =[x_min,xmax,ymin,y_max]
    #输出格式：模型需要的.pkl 方式
    #overpass_shp =[x_min,ymin,xmax,y_max] 归一到0-1之间了
    
    shp_W =shp[1] -shp[0]
    shp_H =shp[3] -shp[2]
    img_W =img[1] -img[0]
    img_H =img[3] -img[2]
    
    shp_center_x =(shp[1] +shp[0])/2
    shp_center_y =(shp[2] +shp[3])/2
    img_center_x =(img[1] +img[0])/2
    img_center_y =(img[2] +img[3])/2
    
    
    if (abs(img_center_x -shp_center_x) <= (shp_W + img_W)/2) &(abs(img_center_y -shp_center_y) <= (shp_H + img_H)/2):
        #print('相交')
#        overpass_shp =[max(shp[0],img[0]), min(shp[1],img[1]), max(shp[2],img[2]),min(shp[3],img[3]),1]
        overpass_shp =[(max(shp[0],img[0]) -img[0])/cut_size,  (min(shp[3],img[3]) -img[2])/cut_size,
                       (min(shp[1],img[1]) -img[0])/cut_size,  (max(shp[2],img[2]) -img[2])/cut_size, 1]
        
        #print(overpass_shp)
        
        if ((overpass_shp[2] - overpass_shp[0]) >=0.028 and (overpass_shp[3] - overpass_shp[1]) >=0.028 ) or(np.min(overpass_shp) !=0 and np.max(overpass_shp) !=1):
            
            return overpass_shp
 

def make_pkl(geomlist_pix_cood,row, colum):

    # 函数用来 得到 第row, colum处图像的 矩形标签数据
    #  左上角为坐标系 原点
    #  shp_cood 格式已变为 [x_min,ymin,xmax,y_max]， 符合.pkl形式
    img_extent =get_img_extent(row, colum)
    shp_cood =[]
    for shp in geomlist_pix_cood:
        overpass_shp =is_overpass(cut_size, shp, img_extent)
        if overpass_shp is not None:
           
            shp_cood.append(overpass_shp)
    roi_arr =np.array(shp_cood, dtype=np.float64)
    
    return roi_arr

if __name__ == "__main__":
    

    #TODO   裁切参数
    st =time.time()
    image_path = r'C:\Liuxiangyu\dataSet\zy3_3973\zy3_3973.tif'
    shp_path = r'C:\Liuxiangyu\dataSet\zy3_3973\aoi_shp\zy3_label_Envelope.shp'
    picture_path = r'C:\Liuxiangyu\dataSet\zy3_3973\train_pictures'
    pic_baseName = 'zy3973_300_'
    
    bands =[4,3,2]  # 真实波段  最小为1
    cut_size =300   # 图像才切成的 大小
    overlap =150    # 裁切时 重叠范围
    if len(os.listdir(picture_path)) > 0:
        print('文件夹 %s 已包含图片，\n是否覆盖?  请选择y/n?'%picture_path)
        inpu =input()
        if inpu =='y':
            pass
        else:
            raise('重设文件夹')
        
        
    dataset, im_width, im_height, x0, y0, x_pixel, y_pixel =read_img(image_path, overlap)
    rows, colums =int(im_height/overlap), int(im_width/overlap)
    
#    im_pic,glo_min,glo_max =need_to_cut(tif_data, bands)
    num =1
    geomlist_pix_cood= readShap(shp_path, x0, y0, x_pixel, y_pixel)
    #im_pic = srtetch_to_255(im_pic,throd_perscent =0.05)
    #print(glo_min,glo_max)
    roi_dict ={}
    all_num =rows*colums
    for row in range(rows):
        for colum in range(colums):
            #保存所有照片
            #jpg_name =cut_to_jpg(dataset, row, colum, bands, foder=picture_path, base_name=pic_baseName)
            roi_arr =make_pkl(geomlist_pix_cood,row, colum)
            if len(roi_arr) !=0:
                #只把有样本的图片保存下来
                jpg_name =cut_to_jpg(dataset, row, colum, bands, foder=picture_path, base_name=pic_baseName)
                roi_dict[jpg_name] =roi_arr
                out_info ='######' +'已处理%d 张'%num +'######\r' 
                out_info_0 ='######' +'共%d 张'%all_num +'######'
                sys.stdout.write(out_info)
                sys.stdout.write(out_info_0)
                sys.stdout.flush()
                
                num +=1
            
            
            
    pickle.dump(roi_dict,open(r'C:\Liuxiangyu\dataSet\Zy3_3973\zy3973_300_.pkl', 'wb'))
    print('\n已保存')
    print('用时%d秒'%(time.time() - st))






