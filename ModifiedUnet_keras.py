# 피클파일 불러오기
import pickle

with open("CARdata.pkl", "rb") as f:
  our_images = pickle.load(f)
with open("CARlabel.pkl", "rb") as f:
  our_labels = pickle.load(f)

print(our_images.shape)
print(our_labels.shape)


# 훈련데이터 80%, 시험데이터 20%
Ntrain = int(our_images.shape[0]*0.8)
Ntest = our_images.shape[0] - Ntrain

x_train = our_images[:Ntrain]
t_train = our_labels[:Ntrain]
x_test = our_images[Ntrain:]
t_test = our_labels[Ntrain:]


print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

#데이터 확장
import numpy as np
import skimage.transform as transform

x_train2 = np.zeros((x_train.shape[0]*6, 256,256,3))
t_train2 = np.zeros((t_train.shape[0]*6, 256,256))

for i in range(x_train.shape[0]):
  x_train2[5*i] = x_train[i]
  x_train2[5*i+1] = transform.rotate(x_train[i], angle=90, resize=False)
  x_train2[5*i+2] = transform.rotate(x_train[i], angle=180, resize=False)
  x_train2[5*i+3] = transform.rotate(x_train[i], angle=270, resize=False)
  x_train2[5*i+4] = x_train[i,::-1,:]
  x_train2[5*i+5] = x_train[i,:,::-1]

  t_train2[5*i] = t_train[i]
  t_train2[5*i+1] = transform.rotate(t_train[i], angle=90, resize=False)
  t_train2[5*i+2] = transform.rotate(t_train[i], angle=180, resize=False)
  t_train2[5*i+3] = transform.rotate(t_train[i], angle=270, resize=False)
  t_train2[5*i+4] = t_train[i,::-1,:]
  t_train2[5*i+5] = t_train[i,:,::-1]


x_test2 = np.zeros((x_test.shape[0]*6, 256,256,3))
t_test2 = np.zeros((t_test.shape[0]*6, 256,256))

for i in range(x_test.shape[0]):
  x_test2[5*i] = x_test[i]
  x_test2[5*i+1] = transform.rotate(x_test[i], angle=90, resize=False)
  x_test2[5*i+2] = transform.rotate(x_test[i], angle=180, resize=False)
  x_test2[5*i+3] = transform.rotate(x_test[i], angle=270, resize=False)
  x_test2[5*i+4] = x_test[i,::-1,:]
  x_test2[5*i+5] = x_test[i,:,::-1]

  t_test2[5*i] = t_test[i]
  t_test2[5*i+1] = transform.rotate(t_test[i], angle=90, resize=False)
  t_test2[5*i+2] = transform.rotate(t_test[i], angle=180, resize=False)
  t_test2[5*i+3] = transform.rotate(t_test[i], angle=270, resize=False)
  t_test2[5*i+4] = t_test[i,::-1,:]
  t_test2[5*i+5] = t_test[i,:,::-1]

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, \
  UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


def Encoder(x, filters=64):

  x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.3)(x)

  x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.3)(x)

  x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  pool = MaxPooling2D(pool_size=(2, 2))(x)

  return x, pool


def Decoder(x, _c=None, filters=64):

  if _c != None:
      x = concatenate([x, _c], axis=-1)

  x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.3)(x)

  x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.3)(x)

  x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = UpSampling2D(size=(2, 2))(x)

  return x


def Outblock(x, _c=None, filters=64):

  if _c != None:
      x = concatenate([x, _c], axis=-1)

  x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.3)(x)

  x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.3)(x)

  x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2D(2, (3, 3), kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('softmax')(x)

  return x


def UnetSegmentation(input_size=(256, 256, 3)):
  in1 = Input(shape=input_size)

  inc1, pool1 = Encoder(in1, 64)
  inc2, pool2 = Encoder(pool1, 128)
  inc3, pool3 = Encoder(pool2, 256)
  inc4, pool4 = Encoder(pool3, 512)

  x = Decoder(pool4, None, 1024)
  x = Decoder(x, inc4, 512)
  x = Decoder(x, inc3, 256)
  x = Decoder(x, inc2, 128)

  x = Outblock(x, inc1, 64)

  model = Model(inputs=[in1], outputs=[x])

  return model

# 모델 컴파일
lr = 0.001
mini_batch = 5
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["gpu:0"])
with mirrored_strategy.scope():
  model = UnetSegmentation()
  model.compile(optimizer=Adam(learning_rate=lr), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

mcp_save = tf.keras.callbacks.ModelCheckpoint('unet_param.h5', save_best_only=True, monitor='val_accuracy', mode='max')

hist = model.fit(x_train2, t_train2, batch_size=mini_batch, validation_data=(x_test2, t_test2), epochs=100, callbacks=[mcp_save])

pred_val = np.zeros_like(t_test2)
for i in range(x_test2.shape[0]//mini_batch+1):
  pred_val[mini_batch*i:mini_batch*(i+1)] = np.argmax(model.predict(x_test2[mini_batch*i:mini_batch*(i+1)]), axis=3)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))
plt.subplot(2, 3, 1)
plt.imshow(x_test2[5])
plt.subplot(2, 3, 2)
plt.imshow(t_test2[5])
plt.subplot(2, 3, 3)
plt.imshow(pred_val[5])

plt.subplot(2, 3, 4)
plt.imshow(x_test2[35])
plt.subplot(2, 3, 5)
plt.imshow(t_test2[35])
plt.subplot(2, 3, 6)
plt.imshow(pred_val[35])
