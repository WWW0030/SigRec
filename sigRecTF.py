# importing dependencies
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model


# limiting gpu usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Preprocessed Data
tf.keras.utils.image_dataset_from_directory('data', batch_size = 12)
data = tf.keras.utils.image_dataset_from_directory('data')
data = data.map(lambda x,y: (x/255, y))
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

train_size = int(len(data)*0.7)
val_size = int(len(data)*0.2)
test_size = int(len(data)*0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Setting up model
model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation= 'relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation= 'relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation= 'relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
print(model.summary())

# Train model
logdir='logs'
tensorboard_callback = tf.keras.callbacks,TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val)

# Test model
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy}')
model.save(os.path.join('models', 'sigRec.h5'))
