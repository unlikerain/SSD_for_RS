# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 21:17:14 2018

@author: tm
"""


import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
#from keras.models import Model
#from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf

from ssd_network import SSD300
from ssd_lossFunction import MultiboxLoss
from ssd_utils import BBoxUtility

#%matplotlib inline
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
set_session(tf.Session(config=config))

NUM_CLASSES = 1+1  # 1 means mask
input_shape = (300, 300, 3)




class Generator(object):
    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,#饱和度
                 brightness_var=0.5,#亮度
                 contrast_var=0.5,#对比度
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3./4., 4./3.]):
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range
        
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
    
    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y
    
    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y
    
    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))     
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y+h, x:x+w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])

        return img, new_targets
    
    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:            
                img_path = self.path_prefix + key
                img = imread(img_path).astype('float32')
                y = self.gt[key].copy()
                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                img = imresize(img, self.image_size).astype('float32')
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)                
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets




#for key in gt:
#    newkey =key.split('.')[0]+ '.jpg'
#    gt[newkey] = gt.pop(key)
#    
#for key in gt:
#    newkey =key.split('.')[0]+ '.jpg'
#    gt[newkey] = gt.pop(key)

#TODO 超参数设置         
#----------------------------------------------------------------------------
#linux_prefix  ='/media/tm/OS/Liuxiangyu/try_7/train_jpg1/'
#path_prefix = linux_prefix
gt = pickle.load(open(r'C:\Liuxiangyu\dataSet\zy3_3973\zy3973_300_3189.pkl', 'rb'))
path_prefix = r'C:\Liuxiangyu\dataSet\zy3_3973\train_pictures_300_100\\'  #path to your pic_data 
batch_size =4
epoch =300
base_lr = 4e-3#3e-4
neg_pos_ratio =3.0

base_name ='./checkpoints/zy3973_300_3189_%s_%s_%s'%(str(batch_size),str(epoch),str(base_lr))
weight_name =base_name +'.hdf5'
png_name =base_name +'.png'
txt_name =base_name +'.txt'
csv_name =base_name +'.csv'
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)
keys = sorted(gt.keys())
num_train = int(round(0.7 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)  



gen = Generator(gt, bbox_util, batch_size, path_prefix,
                train_keys, val_keys,
                (input_shape[0], input_shape[1]),
                saturation_var=0.3,#饱和度
                 brightness_var=0.3,#亮度
                 contrast_var=0.3,#对比度
                 lighting_std=0.3, do_crop=False)

model = SSD300(input_shape, num_classes=NUM_CLASSES)

#TODO 加载初始权重 
model.load_weights('./weights_SSD300.hdf5', by_name=True)

freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2']#,
#         'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
#          'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

for L in model.layers:
    if L.name in freeze:
        L.trainable = False
        

def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)
    #return base_lr * decay/(epoch+1)

callbacks = [keras.callbacks.ModelCheckpoint(weight_name,
                                             verbose=1,
                                             save_best_only=True),
             #keras.callbacks.LearningRateScheduler(schedule),
             #keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0),
             keras.callbacks.CSVLogger(csv_name, separator=',', append=False),
             keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto'),
             keras.callbacks.ReduceLROnPlateau( factor=0.7, patience=4, min_lr=0.0000001)
             ]


optim = keras.optimizers.Adam(lr=base_lr)
# optim = keras.optimizers.RMSprop(lr=base_lr)
# optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio).compute_loss)


history = model.fit_generator(gen.generate(True), int(len(train_keys)/batch_size),#len(train_keys)
                              epoch, verbose=1,
                              callbacks=callbacks,
                              validation_data=gen.generate(False),
                              nb_val_samples=gen.val_batches,
                              workers=1)


#将超参数记录到txt文件里
f =open(txt_name, "w")

f.write('实验的相关参数：\n')
f.write('\n')
f.write('实验所用数据为：zy3973, 用了其432波段,没有进行随机裁剪')
f.write('batch_size =%d\n'%batch_size)
f.write('epoch = %d\n'%epoch)
f.write('base_lr = %s\n'%str(base_lr))
f.write('neg_pos_ratio = %s\n'%str(neg_pos_ratio))
f.write('以下网络层参数被冻结，未参与训练：\n')
f.write('\n')
for lay in freeze:
    f.write(lay+'\n')
f.close()

# 画出损失曲线并保存
fig,ax = plt.subplots() 
plt.xlabel('epoch')  
plt.ylabel('loss')  

plt.plot(history.history['val_loss'],label="val_loss")
plt.plot(history.history['loss'],label="loss")

plt.grid(True)  
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)  


plt.savefig(png_name)
plt.show()  















