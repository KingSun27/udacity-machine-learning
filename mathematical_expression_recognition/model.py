from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM, Conv2D, MaxPooling2D, GlobalAveragePooling2D,Flatten,BatchNormalization,Activation,Reshape,TimeDistributed


def get_model():

	model = Sequential()

	# IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64,300,1)))
	model.add(Conv2D(32, (3, 3)))
#	model.add(Dropout(0.4))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
#	model.add(Dropout(0.4))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(Conv2D(128, (3, 3)))
#	model.add(Dropout(0.4))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(Conv2D(256, (3, 3)))
#	model.add(Dropout(0.4))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(Conv2D(1100, (3, 3), activation='relu'))
	model.add(GlobalAveragePooling2D())
	model.add(Reshape((11,100)))

	model.add(CuDNNLSTM(128, return_sequences=True))
#	model.add(Dropout(0.4))

	model.add(CuDNNLSTM(128, return_sequences=True))
#	model.add(Dropout(0.4))

	model.add(CuDNNLSTM(256, return_sequences=True))
#	model.add(Dropout(0.4))

	model.add(CuDNNLSTM(256, return_sequences=True))
#	model.add(Dropout(0.4))

	model.add(TimeDistributed(Dense(17)))
	model.add(Activation('softmax'))

	return model

