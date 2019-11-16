# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 09:58:35 2019

@author: max
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import pickle
    

infile = open('train_test.pickle','rb')
data = pickle.load(infile)

X_train, X_val, y_train, y_val = data['X_train'],data['X_val'],data['y_train'],data['y_val']


import time 
warnings.filterwarnings('ignore')

# Thời gian bắt đầu: 
start_time = time.time()
numOfEpoch = 250

# 5. Định nghĩa model
model = Sequential()
# Thêm Convolutional layer với 36 kernel, kích thước kernel 3*3
# dùng hàm sigmoid làm activation và chỉ rõ input_shape cho layer đầu tiên
model.add(Conv2D(48, (3, 3), activation='relu', input_shape=(48,33,1)))
# Thêm Convolutional layer
model.add(Conv2D(48, (3, 3), activation='relu'))
# Thêm Max pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
# Flatten layer chuyển từ tensor sang vector
model.add(Flatten())
# Thêm Fully Connected layer với 128 nodes và dùng hàm sigmoid
model.add(Dense(128, activation='sigmoid'))
# Output layer với 10 node và dùng softmax function để chuyển sang xác xuất.
model.add(Dense(36, activation='softmax'))

    
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# 7. Thực hiện train model với data
H = model.fit(X_train, y_train, validation_data=(X_val, y_val),
          batch_size=36, epochs=numOfEpoch, verbose=1)


print('Training Time:', str(time.time() - start_time))


# 8. Vẽ đồ thị loss, accuracy của traning set và validation set

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(18,7))

ax1.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
ax1.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
ax2.plot(np.arange(0, numOfEpoch), H.history['acc'], label='accuracy')
ax2.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')
ax1.set_title('Loss')
ax2.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax2.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')
plt.legend()

# 9. SAVE

model.save("model_captcha.h5")