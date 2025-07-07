# PROJECT 5
import tensorflow as tf
import time
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.datasets import cifar10

# MARK: Simple CNN for baseline
def train_cnn(X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
    # Start timing
    start_time = time.time()
    
    # define CNN architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), # 32 filters, 3x3 kernel
        tf.keras.layers.MaxPooling2D((2, 2)), # 2x2 pooling layer to reduce spatial dimensions
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # 64 filters, 3x3 kernel
        tf.keras.layers.MaxPooling2D((2, 2)), # 2x2 pooling layer
        tf.keras.layers.Flatten(), # Flatten feature maps
        tf.keras.layers.Dense(128, activation='relu'), # Dense layer with 128 neurons
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
    ])

    # model training parameters
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', # sparse labels
                  metrics=['accuracy'])

    # train the model w/ validation split
    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, validation_split=0.1, verbose=2) # verbose=2 for less output

    # Stop timing
    elapsed_time = time.time() - start_time
    print(f"Simple CNN Training Time: {elapsed_time:.2f} seconds")

    # eval model performance on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Simple CNN Test Accuracy: {test_acc:.4f}")

    # genarate predictions and classification report
    y_pred = model.predict(X_test, verbose=2) # get raw predictions
    y_pred_labels = tf.argmax(y_pred, axis=1).numpy() # convert to labels

    print("\nSimple CNN Classification Report:")
    print(classification_report(y_test, y_pred_labels))

    # return predictions and loss history for comparison
    return y_pred_labels, history.history['loss'], history.history['val_loss'] # return train and val loss

# MARK: RFM
import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# visual and progress tracking
import matplotlib.pyplot as plt
from tqdm import tqdm
# track loss during LR training
def track_lr_loss(X_train, y_train, X_val, y_val, pipeline, n_checkpoints=10): 
    train_losses = [] 
    val_losses = []
    chunk_size = len(X_train) // n_checkpoints

    start_time = time.time()
    for i in tqdm(range(n_checkpoints), desc="Training checkpoints"):
        checkpoint_start = time.time()
        
        X_subset = X_train[:chunk_size * (i + 1)] # get subset of training data
        y_subset = y_train[:chunk_size * (i + 1)] # get subset of training labels

        pipeline.fit(X_subset, y_subset)

        y_train_pred = pipeline.predict_proba(X_subset) # get predictions
        train_loss = log_loss(y_subset, y_train_pred) # calc loss
        train_losses.append(train_loss) # store loss

        y_val_pred = pipeline.predict_proba(X_val) # get validation predictions
        val_loss = log_loss(y_val, y_val_pred) # calc validation loss
        val_losses.append(val_loss) # store validation loss

        print(f"\nCheckpoint {i+1}/{n_checkpoints}")
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Elapsed time for checkpoint {i+1}: {time.time() - checkpoint_start:.2f} seconds")

    print(f"Total time for RFM training: {time.time() - start_time:.2f} seconds")
    return train_losses, val_losses

    # trains a random feature model with the given parameters
def trainRFM(X_train, y_train, X_test, y_test,
             method='rbf', n_components=2000,
             gamma='scale', n_checkpoints=10):
    # Start timing
    start_time = time.time()

    # flatten images
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # split training data to get validation set
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_flat, y_train, test_size=0.2, random_state=42
    )

    # feature approx based on param
    feature_map = RBFSampler(n_components=n_components, gamma=gamma, random_state=42)

    # pipeline with preprocessing and LR
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_map', feature_map),
        ('classifier', LogisticRegression(max_iter=300, n_jobs=-1))
    ])

    # track and plot loss curves
    print(f"\nTraining {method.upper()} model with loss tracking...")
    train_losses, val_losses = track_lr_loss(
        X_train_split, y_train_split, X_val, y_val, # training data
        pipeline, n_checkpoints=n_checkpoints # checkpoints
    )

    # Stop timing
    elapsed_time = time.time() - start_time
    print(f"{method.upper()} Random Feature Model Training Time: {elapsed_time:.2f} seconds")

    # final eval on test set
    y_pred = pipeline.predict(X_test_flat)

    print(f"\n{method.upper()} Random Feature Model Results:")
    print(classification_report(y_test, y_pred))        

    return y_pred, train_losses, val_losses

# plot the loss curves
def plot_loss_curves(rfm_train_losses, rfm_val_losses,
                            cnn_train_losses, cnn_val_losses,
                            title="Model Loss Curves Comparison"):
    plt.figure(figsize=(12, 7))

    # plot RFM losses
    checkpoints = range(1, len(rfm_train_losses) + 1)
    plt.plot(checkpoints, rfm_train_losses, 'b-', label='RFM Training Loss')
    plt.plot(checkpoints, rfm_val_losses, 'r-', label='RFM Validation Loss')

    # plot CNN losses - adjust x-axis to match number of checkpoints
    epochs = np.linspace(1, len(rfm_train_losses), len(cnn_train_losses))
    plt.plot(epochs, cnn_train_losses, 'g--', label='CNN Training Loss')
    plt.plot(epochs, cnn_val_losses, 'y--', label='CNN Validation Loss')

    plt.title(title)
    plt.xlabel('Checkpoint/Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# compare CNN & RF model
def compare_models(cnn_preds, rf_preds, y_test):
    # calc metrics for both models
    models = {
        'CNN': cnn_preds,
        'Random Features': rf_preds
    }

    metrics = {}
    for name, preds in models.items(): # iterate over models
        metrics[name] = { # store metrics
            'accuracy': accuracy_score(y_test, preds), # calc accuracy
            'precision': precision_score(y_test, preds, average='weighted'), # calc precision
            'recall': recall_score(y_test, preds, average='weighted'), # calc recall
            'f1': f1_score(y_test, preds, average='weighted') # calc f1
        }

    # create comparison visualizations
    plt.figure(figsize=(12, 6)) 

    # bar plot comparing metrics
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metric_names))
    width = 0.35

    plt.bar(x - width/2, [metrics['CNN'][m] for m in metric_names], width, label='CNN') 
    plt.bar(x + width/2, [metrics['Random Features'][m] for m in metric_names], width, label='Random Features')

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison: CNN vs Random Features')
    plt.xticks(x, metric_names)
    plt.legend()
    plt.grid(True, alpha=0.3) 
    plt.show()

    # detailed comparison
    print("\nDetailed Model Comparison:")
    print("-" * 50) 
    for name, model_metrics in metrics.items():
        print(f"\n{name} Model:")
        for metric, value in model_metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    # load and preprocess data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # train CNN
    cnn_preds, cnn_train_losses, cnn_val_losses = train_cnn(
        X_train, y_train, X_test, y_test, epochs=10, batch_size=64 # params
    )

    # train RFM
    rbf_preds, rbf_train_losses, rbf_val_losses = trainRFM(
        X_train, y_train, X_test, y_test,  # data
        method='rbf', n_components=16000, n_checkpoints=10 # params
    )

    # loss curves
    plot_loss_curves(
        rbf_train_losses, rbf_val_losses, # RFM losses
        cnn_train_losses, cnn_val_losses, # CNN losses
        title="CNN vs RFM Loss Curves"
    )

    # other comparisons
    compare_models(cnn_preds, rbf_preds, y_test)

