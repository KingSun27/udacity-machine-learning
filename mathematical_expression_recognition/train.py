import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint,History,Callback  
from sklearn.model_selection import train_test_split
from data import get_data
from model import get_model
'''
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('lacc'))
'''

# step 1 整理数据
X,y = get_data(70000)
X_train=X[:65000]
X_test=X[65000:]
y_train=y[:65000]
y_test=y[65000:]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


#print(X_train[0])

#step 2 确定模型
model_version = '60000data_lr001_dropout_02_new3'
model = get_model()
model.summary()
model.load_weights('saved_models/weights.best.60000data_lr001_dropout_02_new2.hdf5')

opt = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6)

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.{}.hdf5'.format(model_version), 
                               verbose=1, save_best_only=True)
# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)
print(model.evaluate(X_test, y_test))

history = model.fit(X_train,y_train,
	epochs=20,
	batch_size=128,
	callbacks=[checkpointer],
	validation_data=(X_test, y_test))

#print(history.losses)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./history/{}_accuracy.png'.format(model_version))

# summarize history for loss
plt.cla()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./history/{}_loss.png'.format(model_version))