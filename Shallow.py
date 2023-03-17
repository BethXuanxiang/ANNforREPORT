import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
print(tf.__version__)
print(keras.__version__)
 
# Load and check the datasets:
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, Y_test)=fashion_mnist.load_data()
print(X_train_full.shape, X_train_full.dtype)

# Normalize the pixel values:
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
Y_valid, Y_train = y_train_full[:5000], y_train_full[5000:]

#Labelling:
class_names=["T_shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "shirt","Sneaker"
             , "Bag", "Anlle boot"]
print(class_names[Y_train[0]])
 
# Define the model architecture
model = tf.keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))# Ten distributions

#Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

#Train the model
history=model.fit(X_train, Y_train, epochs=10, validation_data=(X_valid,Y_valid))
print(history)

# Plot
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
 
# Evaluate the model
print(model.evaluate(X_test, Y_test))
