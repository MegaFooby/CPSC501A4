#For Google Collab
#try:
#  # %tensorflow_version only exists in Colab.
#  %tensorflow_version 2.x
#except Exception:
#  pass

import tensorflow as tf
import numpy as np

print("--Get data--")
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

print("--Process data--")
print(len(y_train))
x_train, x_test = x_train / 255.0, x_test / 255.0
 
print("--Make model--")
model = tf.keras.models.Sequential([
  tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
  tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2), strides=2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2), strides=2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(x_train, y_train, epochs=10, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

model.save('notMNIST.h5')
