import random
import numpy as np
import json
import tensorflow as tf
from datagen import DataGen
from sklearn.model_selection import train_test_split


# Engage the pain train
print("HERE COMES THE PAIN TRAIN!")
print()

# Load the training data
print("Loading training data...")

training_data = DataGen().load_data("training_data.json")

print("Training data loaded.")

random.shuffle(training_data)

X = np.array([item[1] for item in training_data])
y = np.array([item[0] for item in training_data])

print("X (features) info: " + str(X.shape) + " " + str(X.dtype))
print("y (labels) info: " + str(y.shape) + " " + str(y.dtype))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the neural network model
print("Defining the model...")

# How did I get those hyperparameters? Tbh I dunno, I just thought "we need
# this many input and this many output", and God guided me the rest of the way 🙏
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(128,)),  # input layer (1)
        tf.keras.layers.Dense(128, activation="relu"),  # hidden layer (2)
        tf.keras.layers.Dense(64, activation="relu"),  # hidden layer (3)
        tf.keras.layers.Dense(32, activation="relu"),  # hidden layer (4)
        tf.keras.layers.Dense(16, activation="relu"),  # hidden layer (5)
        tf.keras.layers.Dense(3, activation="softmax")  # output layer (6)
    ]
)

print("Compiling the model...")

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
print("Training the model...")

model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model
print("Evaluating the model...")

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.2f}")

# Save the trained model
print("Saving the model...")

model.save("obliterator.keras")
