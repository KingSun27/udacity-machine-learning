import tensorflow as tf
import cv2
import numpy as np
from model import get_model
from data import get_data


X,y = get_data(nrows = 5000,skiprows=95000,isReturnImageArray=True)


model = get_model()
model.load_weights('saved_models/weights.best.alldata_lr00001_v3.hdf5')

pre = model.predict(X)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)
print(model.evaluate(X,y))
print(pre.shape)

def get_onehot_from_softmax(labels):
	onehot_arr = []
	for label in labels:
		the_label = np.zeros(labels[0].shape)
		for i,c_onehot in enumerate(label):
			the_label[i][np.argmax(c_onehot)] = 1
		onehot_arr.append(the_label)
	return np.array(onehot_arr)

def get_expressions_from_onehot(onehot):
	CHARS = ['0','1','2','3','4','5','6','7','8','9','+','-','*','/','(',')','=','']
	expressions = []
	for label in onehot:
		expression = ''
		for c in label:
			expression += CHARS[np.argmax(c)]
		expressions.append(expression)
	return expressions


pre_onehot = get_onehot_from_softmax(pre)
print(pre_onehot.shape)
#cv2.imshow('image',X[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
y_expression = get_expressions_from_onehot(y)
y_pre_expression = get_expressions_from_onehot(get_onehot_from_softmax(pre))

right = 0
wrong = []
for i,_ in enumerate(y_expression):
    if y_expression[i] == y_pre_expression[i]:
        right += 1
    else:
        wrong.append(i)
print(right)
print(right/len(y_expression))
