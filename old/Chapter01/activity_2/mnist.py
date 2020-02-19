#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense, Dropout
import datetime

# In[17]:


mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0


# In[20]:

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], X_test.shape[2],1)


# In[21]:


model = Sequential()
model.add(Convolution2D(filters = 10, kernel_size = 3, input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))


# In[22]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[23]:


model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test),callbacks=[tensorboard_callback])


# In[ ]:




