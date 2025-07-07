# MARK: imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# MARK: 1. Data Preprocessing
print("Mark 1")
# load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32') / 255.0 
y = mnist.target.astype('int')

# limiting to 10000 samples
X = X[:10000]
y = y[:10000]

# split the dataset into training and testing sets (80/20 split)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train_full)}")
print(f"Test set size: {len(X_test)}")

# MARK: 2. Model Implementation
print("Mark 2")
# init models
knn = KNeighborsClassifier()
lr = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
svm = SVC(probability=True)

# MARK: 3. Hyperparameter Tuning
print("Mark 3")
# KNN: determine optimal k using ten-fold cross-validation
k_values = range(1, 11)
knn_scores = []

for k in k_values:
    print(f"k={k}")
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = []
    # CV performed 10 times with different seeds
    for seed in range(10):
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        cv_scores = cross_val_score(knn_model, X_train_full, y_train_full, cv=kf)
        scores.append(cv_scores.mean())
    knn_scores.append(np.mean(scores))

optimal_k = k_values[np.argmax(knn_scores)]
print(f"Optimal # neighbors for KNN: {optimal_k}")

# now train KNN w/ optimal k
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train_full, y_train_full)

# SVM: testing with different kernels & regularization parameters
print("\nSVM")
# using smaller set bc it takes so long
X_train_subset = X_train_full[:1000] 
y_train_subset = y_train_full[:1000]

svm_params = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
}

print("Grid search")
svm_grid = GridSearchCV(SVC(probability=True), svm_params, cv=3)  
svm_grid.fit(X_train_subset, y_train_subset)
best_params = svm_grid.best_params_
print(f"Best SVM params: {best_params}")

# train SVM w/ best parameters
print("Training SVM")
svm = SVC(**best_params, probability=True)
svm.fit(X_train_full, y_train_full)

# LR: hyperparam tuning
print("\LR grid search")
lr_params = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
}

lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=5)
lr_grid.fit(X_train_full, y_train_full)
best_lr_params = lr_grid.best_params_
print(f"Best LR params: {best_lr_params}")

# train LR with best parameters
print("Training LR")
lr = LogisticRegression(**best_lr_params, max_iter=1000)
lr.fit(X_train_full, y_train_full)

# MARK: 4. Performance Evaluation
print("\nEvaluating")
models = {'KNN': knn, 'Logistic Regression': lr, 'SVM': svm}
metrics = {}

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    metrics[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

    # confusion matrix
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

# MARK: 5. Comparative Analysis
print("\nFinal Results:")
print("=" * 50)
for model_name, metric in metrics.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {metric['accuracy']:.4f}")
    print(f"Precision: {metric['precision']:.4f}")
    print(f"Recall: {metric['recall']:.4f}")
    print(f"F1 Score: {metric['f1_score']:.4f}")
    print("-" * 30)
