import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Function to ensure directories exist
def create_directory(directory): # Create a directory if it does not exist
    if not os.path.exists(directory): 
        os.makedirs(directory)

# Save plot to a specified folder
def save_plot(folder, filename):
    create_directory(folder)
    plt.savefig(os.path.join(folder, filename))

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize data
y_train = y_train.flatten() # Flatten labels
y_test = y_test.flatten() # Flatten labels

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}") 
print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

# Define a function to plot metrics and save to folder
def plot_metrics(history, title="Model Performance", folder="plots", filename="metrics.png"): # Plot accuracy and loss metrics
    plt.figure(figsize=(12, 5)) # Set figure size

    # Plot accuracy
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st subplot
    plt.plot(history.history['accuracy'], label="Training Accuracy") # Training accuracy
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy") # Validation accuracy
    plt.title(f"{title} - Accuracy") # Title
    plt.xlabel("Epochs") # X-axis label
    plt.ylabel("Accuracy") # Y-axis label
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label="Training Loss") # Training loss
    plt.plot(history.history['val_loss'], label="Validation Loss") # Validation loss
    plt.title(f"{title} - Loss") # Title
    plt.xlabel("Epochs") # X-axis label
    plt.ylabel("Loss") # Y-axis label
    plt.legend()

    plt.tight_layout()
    save_plot(folder, filename)
    plt.show()

# Baseline CNN Model
def build_baseline_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), # 32 filters, 3x3 kernel
        layers.MaxPooling2D((2, 2)), # 2x2 pooling layer to reduce spatial dimensions
        layers.Conv2D(64, (3, 3), activation='relu'), # 64 filters, 3x3 kernel
        layers.MaxPooling2D((2, 2)), # 2x2 pooling layer
        layers.Flatten(), # Flatten the feature maps
        layers.Dense(128, activation='relu'), # Dense layer with 128 neurons
        layers.Dropout(0.5),  # Added dropout for better generalization
        layers.Dense(10, activation='softmax')  # 10 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate function
def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
    start_time = time.time()
    history = model.fit(
        X_train, y_train, # Training data
        validation_data=(X_test, y_test), # Use test data for validation
        epochs=epochs, # Number of epochs
        batch_size=batch_size, # Batch size for training
        verbose=1 # Progress bar for training
    )
    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return history

# ResNet50 Model
def build_resnet():
    base_model = ResNet50(
        include_top=False, # Do not include the top layer
        weights=None,  # No pre-trained weights
        input_shape=(32, 32, 3) # CIFAR-10 images size
    )
    model = models.Sequential([
        base_model, # Add base model
        layers.GlobalAveragePooling2D(), # Spatial data reduction
        layers.Dense(10, activation='softmax') # 10 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # Compile model
    return model

# DenseNet121 Model
def build_densenet():
    base_model = DenseNet121(
        include_top=False, # Do not include the top layer
        weights=None,  # No pre-trained weights
        input_shape=(32, 32, 3) # CIFAR-10 images size
    )
    model = models.Sequential([
        base_model, # Add base model
        layers.GlobalAveragePooling2D(), # Global average pooling layer for spatial data reduction
        layers.Dense(10, activation='softmax') # Dense layer for output classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # Compile model
    return model

# Main execution block
if __name__ == "__main__":
    print("\nTraining Baseline CNN...")
    baseline_cnn = build_baseline_cnn()
    baseline_history = train_and_evaluate(baseline_cnn, X_train, y_train, X_test, y_test)
    plot_metrics(baseline_history, title="Baseline CNN Performance", folder="Baseline_CNN", filename="baseline_cnn_metrics.png")

    print("\nTraining ResNet50 Model...")
    resnet_model = build_resnet()
    resnet_history = train_and_evaluate(resnet_model, X_train, y_train, X_test, y_test)
    plot_metrics(resnet_history, title="ResNet50 Performance", folder="ResNet50", filename="resnet50_metrics.png")

    print("\nTraining DenseNet121 Model...")
    densenet_model = build_densenet()
    densenet_history = train_and_evaluate(densenet_model, X_train, y_train, X_test, y_test)
    plot_metrics(densenet_history, title="DenseNet121 Performance", folder="DenseNet121", filename="densenet121_metrics.png")
