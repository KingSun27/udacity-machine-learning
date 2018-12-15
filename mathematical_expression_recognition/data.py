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

def get_data(nrows = 1000,skiprows=0):

	train_df = pd.read_csv(
		'train.csv',
		nrows=nrows,
		skiprows=skiprows,
		names=['filename','label'],header=0
		)

	
	X = np.zeros((nrows,64,300,1))
	for i,path in tqdm(enumerate(train_df['filename'])):
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		x = np.array(img)
		x = x/255.0
		x = x.reshape(64,300,1)
		X[i] = x


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



#get_data(20)