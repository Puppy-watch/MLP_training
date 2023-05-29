import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import IPython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

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

train_xy = pd.read_csv('train_data1.csv')
test_xy = pd.read_csv('test_data1.csv')

x_train, y_train = split_data(train_xy)
x_test, y_test = split_data(test_xy)

def model_builder(hp):
  model = keras.Sequential()

  # Tune the number of units in the first Dense layer
  hp_units = hp.Int('units', min_value = 32, max_value = 1024, step = 32)
  # hp_units1 = hp.Int('units1', min_value=32, max_value=1024, step=32)
  model.add(keras.layers.Dense(input_dim = 120, units = hp_units, activation = 'relu'))
  # model.add(keras.layers.Dense(units=hp_units1, activation='relu'))
  model.add(keras.layers.Dense(8, activation="softmax"))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values = [1e-1, 1e-2, 1e-3, 1e-4])
  hp_momentum = hp.Choice('momentum', values=[0.6, 0.7, 0.8, 0.9])

  model.compile(optimizer = keras.optimizers.SGD(learning_rate = hp_learning_rate, momentum = hp_momentum),
                loss = "sparse_categorical_crossentropy",
                metrics = ['accuracy'])

  return model

tuner = kt.Hyperband(model_builder,
                     objective = 'val_accuracy',
                     max_epochs = 1000,
                     factor = 10,
                     directory = 'my_dir',
                     project_name = 'intro_to_kt')

class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)

tuner.search(x_train, y_train, epochs = 1000, validation_data = (x_test, y_test), callbacks = [ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')} and momentum is {best_hps.get('momentum')}.
""")

model = tuner.hypermodel.build(best_hps)
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=1000, batch_size=32,  verbose=1)


model.save('tuner_all.h5')

(loss, accuracy) = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('tuner.tflite', 'wb') as f:
  f.write(tflite_model)

y_pred_train = model.predict(x_train)

max_y_pred_train = np.argmax(y_pred_train, axis=1)



plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.legend(['train', 'validation'])

# confusion matrix
LABELS = ['0',
          '1',
          '2',
          '3',
          '4',
          '5',
          '6',
          '7']
y_pred_test = model.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

matrix = metrics.confusion_matrix(max_y_test, max_y_pred_test)
plt.figure(figsize=(6, 4))
sns.heatmap(matrix,
            cmap='PuOr',
            linecolor='white',
            linewidths=1,
            xticklabels=LABELS,
            yticklabels=LABELS,
            annot=True,
            fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()