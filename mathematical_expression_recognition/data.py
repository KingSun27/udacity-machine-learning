import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from tqdm import tqdm
from sklearn.model_selection import train_test_split


# step 1 整理数据

def onehot(size,n):
	onehot = np.zeros(size)
	onehot[n] = 1
	return onehot

def get_data(nrows = 100000,skiprows=0,isReturnImageArray=False):
	'''
	参数：
		nrows:读取的行数
		skiprows:跳过的行数
		isReturnImageArray:X返回格式是图片路径还是图片数组，默认False是返回图片路径
	返回：
		X：图片路径或者图片数组（取决于isReturnImageArray）
		y：
	'''
	train_df = pd.read_csv(
		'train.csv',
		nrows=nrows,
		skiprows=skiprows,
		names=['filename','label'],header=0
		)

	X = train_df['filename']

	if isReturnImageArray:
		X = get_im_cv2(X)

	# print(max([len(label) for label in  train_df['label']]))
	# label中最长的是11位
	# 0~9 = +-* () 空 一共17种符号
	# 所以把label转化成11*17的数组,如果label不足11位，则补充空''，直到11位为止。
	CHARS = ['0','1','2','3','4','5','6','7','8','9','+','-','*','(',')','=','']

	y = []

	for label in train_df['label']:	
		label_list = [c for c in label]+["" for i in range(11-len(label))]
		arr = []
		for c in label_list:
			arr.append(onehot(len(CHARS),CHARS.index(c)))	
		y.append(arr)

	#shape (n,11,17)
	y = np.array(y)
	
	print(X.shape)
	print(y.shape)

	return X,y

def get_im_cv2(paths):
	X = np.zeros((len(paths),64,300,1))
	for i,path in enumerate(paths):
	#for i,path in tqdm(enumerate(paths)):
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		x = np.array(img)
		x = x/255.0 #归一化
		x = x.reshape(64,300,1)
		X[i] = x
	return X

def get_batch(X_paths,y,batch_size):
	while 1:
		for i in range(0,len(X_paths),batch_size):
			X_batch = get_im_cv2(X_paths[i:i+batch_size])
			y_batch = y[i:i+batch_size]
			yield({'conv2d_input':X_batch},{'activation_4':y_batch})


#X,y = get_data()
#print(X[:10])