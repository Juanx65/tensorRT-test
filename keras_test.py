import numpy as np
import tensorflow as tf
from models.mnist_resnet_keras import resnet18_mnist

# Cargar y preparar el dataset MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train, axis=-1) / 255.0
x_test = np.expand_dims(x_test, axis=-1) / 255.0

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Crear y compilar el modelo ResNet-18 adaptado para MNIST
input_shape = (28, 28, 1)
num_classes = 10
model = resnet18_mnist(input_shape, num_classes)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
batch_size = 128
epochs = 20
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

model.save("weights/resnet18_mnist.h5")

# Evaluar el modelo
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
