import numpy as np
import tensorflow as tf
import pandas as pd

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

    total_value = np.array(total_value, dtype=np.float32)

    return total_value, label

test_xy = pd.read_csv('./data/test_data_ddu_plus.csv')
x_test, y_test = split_data(test_xy)
print(x_test[0])
print(x_test.shape)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/ddu_plus_layer2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Test the model on random input data.
interpreter.resize_tensor_input(input_details[0]['index'], x_test.shape)
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], x_test)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
output_data = np.argmax(output_data, axis =-1) # 확률 가장 높은 레이블 번호 얻기
print(output_data)
cnt = 0
for i, pred in enumerate(output_data):
    if (pred == y_test[i]):
        cnt += 1

print(cnt) # 비교결과 정답을 맞춘 수
print(cnt/len(y_test)*100) #정확도

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, output_data)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(8)))
disp.plot(cmap=plt.cm.Blues)
plt.show()

