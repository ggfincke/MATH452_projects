# MATH 452 Class Project Showcase

This repository contains a comprehensive collection of machine learning projects completed for **MATH 452: Mathematical Foundations of Machine Learning** at Penn State (Fall 2025). These projects demonstrate practical implementation of core ML algorithms, from traditional methods to modern deep learning architectures, showcasing both theoretical understanding and practical implementation skills.

Achieved a grade of 100% on all projects below.

---

## 🧠 Midterm Projects: Traditional Machine Learning
📁 [`Midterm`](./Midterm)  
Comprehensive exploration of classical machine learning algorithms with hands-on implementation and evaluation.

### 🎯 **Project 1: Multi-Algorithm Classification**
**File:** `FinckeMidterm1.py`  
**Dataset:** MNIST (10,000 samples)  
**Objective:** Compare performance of three classical ML algorithms

**Key Features:**
- **K-Nearest Neighbors (KNN)** with optimal k selection via 10-fold cross-validation
- **Logistic Regression** with L2 regularization and hyperparameter tuning
- **Support Vector Machine (SVM)** with linear and RBF kernels
- Comprehensive performance evaluation using accuracy, precision, recall, and F1-score
- Confusion matrix visualization for each algorithm

**Technical Highlights:**
- Stratified train-test split (80/20)
- Grid search cross-validation for hyperparameter optimization
- Multiple random seeds for robust k-selection in KNN
- Weighted metrics for multi-class evaluation

### 🎯 **Project 2: Unsupervised Learning & Clustering**
**File:** `FinckeMidterm2.py`  
**Dataset:** MNIST (10,000 samples)  
**Objective:** Implement K-Means clustering with dimensionality reduction

**Key Features:**
- **K-Means clustering** with elbow method for optimal cluster selection
- **Silhouette score analysis** for cluster quality assessment
- **Principal Component Analysis (PCA)** for 2D visualization
- Cluster center visualization as reconstructed digit images
- Interactive plotting for cluster analysis

**Technical Highlights:**
- Systematic k-value testing (k=2 to 15)
- Dual evaluation criteria: inertia and silhouette scores
- PCA-based dimensionality reduction for visualization
- Cluster interpretation through centroid analysis

---

## 🤖 Final Projects: Deep Learning & Advanced Methods
📁 [`Final`](./Final)  
Advanced machine learning projects focusing on deep learning architectures and comparative analysis.

**Project Team:** Jacob Goulet, Tyler Rossi, Diego Bueno, Javier Pozo Miranda, Duong Bao, Garrett Fincke

### 🎯 **Project 1: Deep Learning Architecture Comparison**
**File:** `Project1.py`  
**Dataset:** CIFAR-10  
**Objective:** Compare performance of different CNN architectures

**Key Features:**
- **Baseline CNN** - Custom sequential architecture
- **ResNet50** - Residual network implementation
- **DenseNet121** - Densely connected network
- Performance visualization and comparative analysis
- Automated plot generation and organization

**Technical Highlights:**
- TensorFlow/Keras implementation
- Global average pooling for feature extraction
- Dropout regularization for generalization
- Training time and accuracy comparison
- Comprehensive metrics visualization

### 🎯 **Project 2: Traditional vs Modern ML Methods**
**File:** `Project2.py`  
**Dataset:** CIFAR-10  
**Objective:** Compare CNN performance against Random Feature Models

**Key Features:**
- **Convolutional Neural Network** baseline implementation
- **Random Feature Models (RFM)** with RBF kernel approximation
- Loss curve tracking and visualization
- Comprehensive performance comparison
- Feature mapping analysis

**Technical Highlights:**
- RBF sampling for feature approximation
- Pipeline-based preprocessing with StandardScaler
- Progressive training with checkpoint analysis
- Multi-metric evaluation (accuracy, precision, recall, F1)
- Visual comparison of training dynamics

### 👥 Collaboration

Project 2 was primarily implemented by me, Garrett Fincke. Project 1 and supporting analyses were completed collaboratively by:
- Jacob Goulet
- Tyler Rossi
- Diego Bueno
- Javier Pozo Miranda
- Duong Bao

---

## 🛠️ Technical Implementation

**Libraries & Tools:**
- **Core ML:** scikit-learn, TensorFlow/Keras
- **Data Processing:** NumPy, pandas
- **Visualization:** matplotlib, seaborn
- **Evaluation:** sklearn.metrics, confusion matrices

**Key Algorithms Implemented:**
- K-Nearest Neighbors with cross-validation
- Logistic Regression with regularization
- Support Vector Machines (linear/RBF kernels)
- K-Means clustering with elbow method
- Principal Component Analysis
- Convolutional Neural Networks
- ResNet and DenseNet architectures
- Random Feature Models with RBF approximation

**Performance Evaluation:**
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix analysis
- Cross-validation techniques
- Hyperparameter optimization
- Loss curve analysis

---

## 📊 Project Highlights

**Midterm Achievements:**
- Implemented three different classification algorithms from scratch
- Achieved optimal hyperparameter tuning through systematic search
- Demonstrated clustering analysis with visualization
- Applied dimensionality reduction for interpretability

**Final Project Achievements:**
- Built and compared multiple deep learning architectures
- Implemented advanced feature mapping techniques
- Conducted comprehensive performance analysis
- Created automated visualization and reporting systems

---

## 💻 Setup & Usage

**Requirements:**
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn tqdm
```

**Running the Projects:**
```bash
# Midterm Projects
python Midterm/FinckeMidterm1.py  # Classification comparison
python Midterm/FinckeMidterm2.py  # Clustering analysis

# Final Projects  
python Final/Project1.py          # Deep learning architectures
python Final/Project2.py          # CNN vs Random Features
```

**Output:**
- Performance metrics and comparisons
- Confusion matrices and visualizations
- Training curves and loss analysis
- Model architecture summaries

---

## 📚 Course Information
**MATH 452** — Mathematical Foundations of Machine Learning  
**Institution:** Pennsylvania State University  
**Completion Date:** Fall 2023  

**Course Focus:** Theoretical foundations and practical implementation of machine learning algorithms, covering both classical statistical methods and modern deep learning approaches.

---

*These projects demonstrate the progression from traditional machine learning methods to modern deep learning architectures, showcasing both theoretical understanding and practical implementation skills in the field of machine learning.*