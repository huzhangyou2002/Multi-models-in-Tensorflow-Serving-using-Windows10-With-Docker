#Ref https://tensorflow.google.cn/tfx/tutorials/serving/rest_simple

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential([
    keras.layers.Conv2D(input_shape = (28,28,1),filters = 8,kernel_size = 3,strides = 2,activation = 'relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

model.summary()

testing = False
epochs = 5

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(train_images, train_labels, epochs=epochs)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy: {}'.format(test_acc))


MODEL_DIR = "E:\\tmp\\tfserving\\"
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

print('\nSaved model Finished')

#Docker 启动
#docker run -t --rm -p 8501:8501 -v "E:/tmp/tfserving:/models/tfserving" -e MODEL_NAME=tfserving tensorflow/serving
#说明：-v "E:/tmp/tfserving:/models/tfserving" 含义是 将 E:/tmp/tfserving目录 映射为模型及名称/models/tfserving

#构建测试数据
import json
data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

#通过restful api 调用预测
import requests
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/tfserving:predict', data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
for pred in predictions:
    print(class_names[np.argmax(pred)])