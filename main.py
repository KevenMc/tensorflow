import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images,train_labels), (test_images, test_labels) =datasets.cifar10.load_data()

class_names =['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


model= models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(32, (3,3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])

history= model.fit(train_images, train_labels, epochs=10,
validation_data=(test_images,test_labels))

test_loss, test_acc= model.evaluate(test_images, test_labels)

print(test_acc)