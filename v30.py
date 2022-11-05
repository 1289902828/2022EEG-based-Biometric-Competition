




#v30



import scipy.io as scio
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np
from PIL import Image

import paddle
import paddle.nn as nn
from paddle.io import Dataset
import paddle.vision.transforms as T
import paddle.nn.functional as F
from paddle.metric import Accuracy
from sklearn.preprocessing import LabelEncoder
from paddle.optimizer.lr import LinearWarmup
from paddle.optimizer.lr import CosineAnnealingDecay
import warnings
warnings.filterwarnings("ignore")


def seed_everything(seed):
	#random.seed(seed)
	np.random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	paddle.seed(seed)

seed_everything(43)


#版本
date_version = '20221030_v30'
BATCH_SIZE = 1024


# 读取数据
train_images = pd.read_csv('data/data151025/Enrollment_Info.csv')
val_images = pd.read_csv('data/data151025/Calibration_Info.csv')

train_images = shuffle(train_images, random_state=0)
val_images = shuffle(val_images, random_state=0)

# 划分训练集和校验集
train_image_list = train_images
val_image_list = val_images

#df = train_image_list
train_image_path_list = train_image_list['EpochID'].values
train_label_list = train_image_list['SubjectID'].values

val_image_path_list = val_image_list['EpochID'].values
val_label_list = val_image_list['SubjectID'].values





class MyDataset(paddle.io.Dataset):
	
	def __init__(self, train_img_list, val_img_list,train_label_list,val_label_list, mode='train'):
		
		super(MyDataset, self).__init__()
		self.img = []
		self.label = []
		
		self.train_images = train_img_list
		self.test_images = val_img_list
		self.train_label = train_label_list
		self.test_label = val_label_list
		
		if mode == 'train':
			# 读train_images的数据
			for img,la in zip(self.train_images, self.train_label):
				self.img.append('/home/aistudio/data_unzip/train/'+img+'.mat')
				self.label.append(paddle.to_tensor(int(la[4:]) - 1, dtype='int64'))
		else:
			# 读test_images的数据
			for img,la in zip(self.test_images, self.test_label):
				self.img.append('/home/aistudio/data_unzip/val/'+img+'.mat')
				self.label.append(paddle.to_tensor(int(la[4:]) - 1, dtype='int64'))
	
	def load_eeg(self, eeg_path):
		
		data = scio.loadmat(eeg_path)
		
		return data['epoch_data']
	
	def __getitem__(self, index):
		
		eeg_data = self.load_eeg(self.img[index])
		eeg_label = self.label[index]
		# label = paddle.to_tensor(label)
		
		return eeg_data, eeg_label
	
	def __len__(self):
		
		return len(self.img)



#train_loader
train_dataset = MyDataset(train_img_list=train_image_path_list, val_img_list=val_image_path_list, train_label_list=train_label_list, val_label_list=val_label_list, mode='train')
train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

#val_loader
val_dataset = MyDataset(train_img_list=train_image_path_list, val_img_list=val_image_path_list, train_label_list=train_label_list, val_label_list=val_label_list, mode='test')
val_loader = paddle.io.DataLoader(val_dataset, places=paddle.CPUPlace(), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)





import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pywt
from paddle.nn import Linear, Dropout, ReLU
from paddle.nn import Conv1D, BatchNorm1D, MaxPool1D, AvgPool1D
from paddle.nn.initializer import Uniform
from paddle.fluid.param_attr import ParamAttr
from paddle.utils.download import get_weights_path_from_url
from scipy import signal

class MyNet_dwt(nn.Layer):
	
	def __init__(self, num_classes=95):
		super(MyNet_dwt, self).__init__()
		
		self.num_classes = num_classes
		
		
		#self._conv1_1 = Conv1D(64,128,3,stride=2,padding=1)
		#self._conv1_2 = Conv1D(128,256,3,stride=2,padding=1)
		#self._conv1_3 = Conv1D(256,512,3,stride=2,padding=1)
		#self._conv1_4 = Conv1D(512,256,3,stride=2,padding=1)
		
		self._conv2_1 = Conv1D(64,128,3,stride=2,padding=1)
		self._conv2_2 = Conv1D(128,256,3,stride=2,padding=1)
		self._conv2_3 = Conv1D(256,512,3,stride=2,padding=1)
		self._conv2_4 = Conv1D(512,1024,3,stride=2,padding=1)
		self._conv2_5 = Conv1D(1024,1024,3,stride=2,padding=1)
		self._conv2_6 = Conv1D(1024,1024,3,stride=2,padding=1)
		
		self._conv3_1 = Conv1D(64,128,3,stride=2,padding=1)
		self._conv3_2 = Conv1D(128,256,3,stride=2,padding=1)
		self._conv3_3 = Conv1D(256,512,3,stride=2,padding=1)
		self._conv3_4 = Conv1D(512,1024,3,stride=2,padding=1)
		self._conv3_5 = Conv1D(1024,1024,3,stride=2,padding=1)
		self._conv3_6 = Conv1D(1024,1024,3,stride=2,padding=1)
		
		self._conv4_1 = Conv1D(64,128,3,stride=2,padding=1)
		self._conv4_2 = Conv1D(128,256,3,stride=2,padding=1)
		self._conv4_3 = Conv1D(256,512,3,stride=2,padding=1)
		self._conv4_4 = Conv1D(512,1024,3,stride=2,padding=1)
		self._conv4_5 = Conv1D(1024,1024,3,stride=2,padding=1)
		self._conv4_6 = Conv1D(1024,1024,3,stride=2,padding=1)
		
		self._conv5_1 = Conv1D(64,128,3,stride=2,padding=1)
		self._conv5_2 = Conv1D(128,256,3,stride=2,padding=1)
		self._conv5_3 = Conv1D(256,512,3,stride=2,padding=1)
		self._conv5_4 = Conv1D(512,1024,3,stride=2,padding=1)
		self._conv5_5 = Conv1D(1024,1024,3,stride=2,padding=1)
		self._conv5_6 = Conv1D(1024,1024,3,stride=2,padding=1)
		
		#self._conv6_1 = Conv1D(64,128,3,stride=2,padding=1)
		#self._conv6_2 = Conv1D(128,256,3,stride=2,padding=1)
		#self._conv6_3 = Conv1D(256,512,3,stride=2,padding=1)
		#self._conv6_4 = Conv1D(512,256,3,stride=2,padding=1)
		
		#self._bn1_1 = nn.BatchNorm1D(128)
		#self._bn1_2 = nn.BatchNorm1D(256)
		#self._bn1_3 = nn.BatchNorm1D(512)
		#self._bn1_4 = nn.BatchNorm1D(256)
		
		self._bn2_1 = nn.BatchNorm1D(128)
		self._bn2_2 = nn.BatchNorm1D(256)
		self._bn2_3 = nn.BatchNorm1D(512)
		self._bn2_4 = nn.BatchNorm1D(1024)
		self._bn2_5 = nn.BatchNorm1D(1024)
		self._bn2_6 = nn.BatchNorm1D(1024)
		
		self._bn3_1 = nn.BatchNorm1D(128)
		self._bn3_2 = nn.BatchNorm1D(256)
		self._bn3_3 = nn.BatchNorm1D(512)
		self._bn3_4 = nn.BatchNorm1D(1024)
		self._bn3_5 = nn.BatchNorm1D(1024)
		self._bn3_6 = nn.BatchNorm1D(1024)
		
		self._bn4_1 = nn.BatchNorm1D(128)
		self._bn4_2 = nn.BatchNorm1D(256)
		self._bn4_3 = nn.BatchNorm1D(512)
		self._bn4_4 = nn.BatchNorm1D(1024)
		self._bn4_5 = nn.BatchNorm1D(1024)
		self._bn4_6 = nn.BatchNorm1D(1024)
		
		self._bn5_1 = nn.BatchNorm1D(128)
		self._bn5_2 = nn.BatchNorm1D(256)
		self._bn5_3 = nn.BatchNorm1D(512)
		self._bn5_4 = nn.BatchNorm1D(1024)
		self._bn5_5 = nn.BatchNorm1D(1024)
		self._bn5_6 = nn.BatchNorm1D(1024)
		
		#self._bn6_1 = nn.BatchNorm1D(128)
		#self._bn6_2 = nn.BatchNorm1D(256)
		#self._bn6_3 = nn.BatchNorm1D(512)
		#self._bn6_4 = nn.BatchNorm1D(256)
		
		self.avgpool = nn.AvgPool1D(kernel_size=16, stride=1, padding=0)
		
		self._fc8 = Linear(in_features=4109,out_features=num_classes)
	
	def forward(self, inputs):
		
		x_eeg = inputs[:,0:64,:]#x_maker的shape是[batch_size,64,1000]
		x_maker = inputs[:,64,:]#x_maker的shape是[batch_size,1,1000]
		
		#带通滤波
		s = 5
		#b, a = signal.butter(s, 0.032, 'lowpass')
		#fd_4 = signal.filtfilt(b, a, x_eeg)
		b, a = signal.butter(s, [0.032,0.064], 'bandpass')
		fd_8 = signal.filtfilt(b, a, x_eeg)
		b, a = signal.butter(s, [0.064,0.104], 'bandpass')
		fd_13 = signal.filtfilt(b, a, x_eeg)
		b, a = signal.butter(s, [0.104,0.24], 'bandpass')
		fd_30 = signal.filtfilt(b, a, x_eeg)
		b, a = signal.butter(s, [0.24,0.4], 'bandpass')
		fd_50 = signal.filtfilt(b, a, x_eeg)
		#b, a = signal.butter(s, 0.4, 'highpass')
		#fd_250 = signal.filtfilt(b, a, x_eeg)
		
		#x1 = paddle.to_tensor(fd_4, dtype='float32')
		x2 = paddle.to_tensor(fd_8, dtype='float32')
		x3 = paddle.to_tensor(fd_13, dtype='float32')
		x4 = paddle.to_tensor(fd_30, dtype='float32')
		x5 = paddle.to_tensor(fd_50, dtype='float32')
		#x6 = paddle.to_tensor(fd_250, dtype='float32')
		
		
		#卷积
		#x1 = F.relu(self._bn1_1(self._conv1_1(x1)))
		#x1 = F.relu(self._bn1_2(self._conv1_2(x1)))
		#x1 = F.relu(self._bn1_3(self._conv1_3(x1)))
		#x1 = F.relu(self._bn1_4(self._conv1_4(x1)))
		
		x2 = F.relu(self._bn2_1(self._conv2_1(x2)))
		x2 = F.relu(self._bn2_2(self._conv2_2(x2)))
		x2 = F.relu(self._bn2_3(self._conv2_3(x2)))
		x2 = F.relu(self._bn2_4(self._conv2_4(x2)))
		x2 = F.relu(self._bn2_5(self._conv2_5(x2)))
		x2 = F.relu(self._bn2_6(self._conv2_6(x2)))
		
		x3 = F.relu(self._bn3_1(self._conv3_1(x3)))
		x3 = F.relu(self._bn3_2(self._conv3_2(x3)))
		x3 = F.relu(self._bn3_3(self._conv3_3(x3)))
		x3 = F.relu(self._bn3_4(self._conv3_4(x3)))
		x3 = F.relu(self._bn3_5(self._conv3_5(x3)))
		x3 = F.relu(self._bn3_6(self._conv3_6(x3)))
		
		x4 = F.relu(self._bn4_1(self._conv4_1(x4)))
		x4 = F.relu(self._bn4_2(self._conv4_2(x4)))
		x4 = F.relu(self._bn4_3(self._conv4_3(x4)))
		x4 = F.relu(self._bn4_4(self._conv4_4(x4)))
		x4 = F.relu(self._bn4_5(self._conv4_5(x4)))
		x4 = F.relu(self._bn4_6(self._conv4_6(x4)))
		
		x5 = F.relu(self._bn5_1(self._conv5_1(x5)))
		x5 = F.relu(self._bn5_2(self._conv5_2(x5)))
		x5 = F.relu(self._bn5_3(self._conv5_3(x5)))
		x5 = F.relu(self._bn5_4(self._conv5_4(x5)))
		x5 = F.relu(self._bn5_5(self._conv5_5(x5)))
		x5 = F.relu(self._bn5_6(self._conv5_6(x5)))
		
		#x6 = F.relu(self._bn6_1(self._conv6_1(x6)))
		#x6 = F.relu(self._bn6_2(self._conv6_2(x6)))
		#x6 = F.relu(self._bn6_3(self._conv6_3(x6)))
		#x6 = F.relu(self._bn6_4(self._conv6_4(x6)))
		
		#print(x2.shape)
		#[1024,1024,16]
		
		x2 = self.avgpool(x2)
		x3 = self.avgpool(x3)
		x4 = self.avgpool(x4)
		x5 = self.avgpool(x5)
		
		#x_t = paddle.concat(x = [x1, x2, x3, x4, x5, x6], axis=2)
		x_t = paddle.concat(x = [x2, x3, x4, x5], axis=2)
		x_t = paddle.flatten(x_t, start_axis=1, stop_axis=-1)#这里x的shape是[batch_size,]，
		
		#拼上maker
		x_maker = x_maker.squeeze(axis=1)
		#x_maker.shape:[batch_size,1000]
		mk = paddle.max(x_maker, axis=1)
		#mk.shape:[batch_size]
		mk_dense = np.array(np.array(mk) - 1,'int8')
		num_batch = mk_dense.shape[0]
		index_offset = np.arange(num_batch) * 13 #13是测试任务的数量
		mk_onehot = np.zeros((num_batch,13))#batch_size*num_classes
		mk_onehot.flat[index_offset + mk_dense.ravel()] = 1
		mk_onehot = paddle.to_tensor(mk_onehot, dtype='float32')
		x = paddle.concat(x=[x_t,mk_onehot],axis=1)
		#算一下fc8的in_features=  +13
		x = self._fc8(x)
		return x




model_res = MyNet_dwt(num_classes=95)

paddle.summary(model_res,(BATCH_SIZE,65,1000))





# 模型封装
model = paddle.Model(model_res)

# 定义优化器
class Cosine(CosineAnnealingDecay):
	
	def __init__(self, lr, step_each_epoch, epochs, **kwargs):
		super(Cosine, self).__init__(
			learning_rate=lr,
			T_max=step_each_epoch * epochs, )
		
		self.update_specified = False

class CosineWarmup(LinearWarmup):
	
	def __init__(self, lr, step_each_epoch, epochs, warmup_epoch=5, **kwargs):
		assert epochs > warmup_epoch, "total epoch({}) should be larger than warmup_epoch({}) in CosineWarmup.".format(
			epochs, warmup_epoch)
		warmup_step = warmup_epoch * step_each_epoch
		start_lr = 0.0
		end_lr = lr
		lr_sch = Cosine(lr, step_each_epoch, epochs - warmup_epoch)
		
		super(CosineWarmup, self).__init__(
			learning_rate=lr_sch,
			warmup_steps=warmup_step,
			start_lr=start_lr,
			end_lr=end_lr)
		
		self.update_specified = False


s_e_e = int(np.ceil(57851/BATCH_SIZE))
scheduler = CosineWarmup(lr=0.0001, step_each_epoch=s_e_e, epochs=24, warmup_steps=5, start_lr=0, end_lr=0.0001, verbose=True)
optim = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())

# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )
callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir_alexdwt')

use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

# 模型训练与评估
model.fit(train_loader,
        val_loader,
        log_freq=1,
        epochs=24,
        callbacks=callback,
        verbose=1,
        )




model.save('/home/aistudio/model/baseline_'+date_version, True)






# 模型预测并生成提交文件

import os, time
import matplotlib.pyplot as plt
import paddle
from PIL import Image
import numpy as np
import pandas as pd
import scipy.io as scio

use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

param_state_dict = paddle.load('/home/aistudio/model/baseline_'+date_version+'.pdparams')
model_res.set_dict(param_state_dict)
model_res.eval() #训练模式

test_image = pd.read_csv('data/data151025/Testing_Info.csv')
test_image_path_list = test_image['EpochID'].values

eeg_list = list()

labeled_img_list = []
for img in test_image_path_list:
    eeg_list.append('/home/aistudio/data_unzip/test/'+img+'.mat')
    labeled_img_list.append(img)

def load_eeg(eeg_path):
    # 读取数据
    data = scio.loadmat(eeg_path)
    return data['epoch_data']

pre_list = []
for i in range(len(eeg_list)):
    data = load_eeg(eeg_path=eeg_list[i])
    dy_x_data = np.array(data).astype('float32')
    dy_x_data = dy_x_data[np.newaxis,:, :]
    eeg = paddle.to_tensor(dy_x_data)
    out = model_res(eeg)
    res = paddle.nn.functional.softmax(out)[0] # 若模型中已经包含softmax则不用此行代码。
    lab = np.argmax(out.numpy())  #argmax():返回最大数的索引
    pre_list.append(int(lab)+1)

img_test = pd.DataFrame(labeled_img_list)
img_pre = pd.DataFrame(labeled_img_list)

img_test = img_test.rename(columns = {0:"EpochID"})
img_pre['SubjectID'] = pre_list
pre_info = img_pre['SubjectID'].values
test_info = test_image['SubjectID'].values

result_cnn = list()

for i,j in zip(test_info, pre_info):
	
	if i == 'None':
		result_cnn.append(j)
	elif int(i[4:])==j :
		print(i[4:])
		result_cnn.append(int(1))
	else:
		result_cnn.append(int(0))

img_test['Prediction'] = result_cnn




img_test.to_csv('/home/aistudio/submit/result_'+date_version+'.csv', index=False)






