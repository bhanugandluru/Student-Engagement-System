import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_cnn_model
import matplotlib.pyplot as plt
import os
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def train_model():
    dataset_path = 'dataset'
    img_size = (128, 128)
    batch_size = 32
    epochs = 10 # Adjust based on complexity to reach 93%

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    model = create_cnn_model(input_shape=(128, 128, 3), num_classes=4)

    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/model.h5')
    print("Model saved as models/model.h5")

    # Save training graphs
    if not os.path.exists('static/graphs'):
        os.makedirs('static/graphs')

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('static/graphs/accuracy_plot.png')
    plt.close()

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('static/graphs/loss_plot.png')
    plt.close()

    print("Training graphs saved in static/graphs/")

if __name__ == "__main__":
    # Check if dataset exists, if not generate it
    if not os.path.exists('dataset'):
        from data_generation import generate_synthetic_data
        generate_synthetic_data()
        
    train_model()
