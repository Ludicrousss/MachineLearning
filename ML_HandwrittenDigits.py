import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data (scale pixel values to be between 0 and 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into a 784-dimensional vector
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (for digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Function to display a random image and its prediction during training
def display_random_image(epoch, logs):
    index = np.random.randint(0, x_test.shape[0])
    plt.imshow(x_test[index], cmap='gray')
    prediction = model.predict(np.expand_dims(x_test[index], axis=0))
    predicted_label = np.argmax(prediction)
    plt.title(f'Epoch: {epoch + 1}, Predicted: {predicted_label}, Actual: {y_test[index]}')
    plt.axis('off')  # Hide the axis
    plt.draw()
    plt.pause(1)  # Pause for a second to display the image

# Create a custom callback to display random images after each epoch
class LiveView(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        plt.clf()  # Clear the figure
        display_random_image(epoch, logs)

# Ask the user for the number of epochs to train the model
try:
    num_epochs = int(input("Enter the number of epochs for training: "))
except ValueError:
    print("Invalid input. Using default of 5 epochs.")
    num_epochs = 5

# Set up the plot
plt.figure(figsize=(5, 5))

# Train the model with the live view callback
history = model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[LiveView()])

# Final evaluation of the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc}')

# Plotting training and validation loss and accuracy
plt.figure(figsize=(12, 5))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()  # Show the plots

# Wait for user input to keep the window open
input("Press Enter to close the plots and exit...")
