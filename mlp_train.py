from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def split_data(dataset):
    """ 데이터셋 나누기

    Parameters
    ----------
    dataset : 데이터셋

    Returns
    -------
    (데이터, 레이블)
    """

    total_value = []

    dataset = dataset.transpose()
    value = dataset[:-1]
    label = dataset[-1:]
    label = np.array(label.transpose())
    # label = label.reshape(-1)

    for i in range(len(value.columns)):
        v_list = list(np.array(value[i].tolist()))
        total_value.append(v_list)

    total_value = np.array(total_value)

    return total_value, label

train_xy = pd.read_csv('./without_data/train_data_other_plus.csv')
test_xy = pd.read_csv('./without_data/test_data_other_plus.csv')

x_train, y_train = split_data(train_xy)
x_test, y_test = split_data(test_xy)

model = keras.models.Sequential()
model.add(keras.layers.Dense(500, input_dim=120, activation="relu"))
model.add(keras.layers.Dense(7, activation="softmax"))

model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate = 0.1, momentum = 0.6),
              metrics=["accuracy"])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1000, batch_size=32)

plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('layer1 model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('layer1 model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'validation'])
plt.show()

#모델 평가
model.evaluate(x_test, y_test)

X_new = x_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('./without_model/other_plus.tflite', 'wb') as f:
  f.write(tflite_model)