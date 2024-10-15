import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import datetime

# Load the colorectal histology dataset
dataset, info = tfds.load('colorectal_histology', with_info=True, as_supervised=True)

# Split the dataset into training and validation sets
train_dataset = dataset['train']
val_size = int(0.2 * info.splits['train'].num_examples)  # 20% for validation
train_dataset = train_dataset.skip(val_size)  # Skip first val_size examples for training
validation_dataset = dataset['train'].take(val_size)  # Take the first val_size examples for validation

# Preprocessing function
def preprocess_image(image, label):
    image = tf.image.resize(image, [128, 128])  # Resize images to 128x128
    image = image / 255.0  # Normalize pixel values to be between 0 and 1
    return image, label

# Apply preprocessing to the datasets
train_dataset = train_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

# Build the CNN model
num_classes = info.features['label'].num_classes  # Get number of classes from the dataset info
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),  # Define input layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Keep this for integer labels
              metrics=['accuracy'])

# Set up TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Function to display images with predictions
def display_images(images, labels, predictions):
    num_images = len(images)
    max_images = min(num_images, 25)  # Limit to 25 images to avoid subplot error
    cols = 5  # Number of columns in the grid
    rows = (max_images + cols - 1) // cols  # Calculate the number of rows needed

    plt.figure(figsize=(10, 2 * rows))  # Adjust the figure size based on number of rows
    for i in range(max_images):
        plt.subplot(rows, cols, i + 1)  # Dynamic subplot based on the number of images
        plt.imshow(images[i])
        plt.title(f'Pred: {predictions[i]}, True: {labels[i]}')
        plt.axis('off')
    plt.show()
    plt.pause(0.001)  # Pause to allow the plot to update

# Ask the user if they want to see the images after each epoch
show_images = input("Do you want to see the images after each epoch? (yes/no): ").strip().lower() == 'yes'

# Ask the user for the number of epochs
epochs = int(input("Enter the number of epochs for training: "))

# Train the model with TensorBoard callback
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    model.fit(train_dataset, validation_data=validation_dataset, epochs=1, callbacks=[tensorboard_callback])

    # If the user wants to see images, predict on a batch of validation images
    if show_images:
        for images, labels in validation_dataset.take(1):  # Take one batch from validation dataset
            predictions = model.predict(images)
            predicted_classes = np.argmax(predictions, axis=1)
            display_images(images.numpy(), labels.numpy(), predicted_classes)

# Evaluate the model
loss, accuracy = model.evaluate(validation_dataset)
print(f'Validation accuracy: {accuracy * 100:.2f}%')

# Launch TensorBoard (open a terminal or command prompt and run this command)
# tensorboard --logdir logs/fit
