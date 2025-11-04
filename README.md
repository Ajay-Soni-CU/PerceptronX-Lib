# 🚀 IntelliNeuro PerceptronX
https://pypi.org/project/IntelliNeuro/0.1.0/

## 🤖 Custom Perceptron Machine Learning Library

**Author:** Ajay Soni  
**Version:** 0.1.0  
**Repository:** https://github.com/ml-beginner-learner/IntelliNeuro  

---

## 📚 Introduction

**IntelliNeuro’s PerceptronX** is a modular and fully custom-built **Perceptron-based Machine Learning library**, developed from scratch using only **NumPy** and **pandas** for efficient matrix operations and data handling. It’s crafted to give learners and developers a transparent view of how fundamental neural architectures and gradient-based optimizations work internally.

This library supports multiple tasks:
- 🔹 Linear regression (continuous value prediction)
- 🔹 Binary classification using sigmoid activation
- 🔹 Experimental multi-class classification via softmax activation

Designed for clarity and extensibility, **PerceptronX** is ideal for those wanting to *learn, modify, and visualize* how perceptrons evolve through gradient descent and backpropagation in a minimal yet educational setup.

---

## ✨ Key Features

### 🎯 Versatile Learning Capabilities
- Perform **linear regression**, **binary**, and **multi-class classification** from the same unified API.
- Automatically detects the task type based on target variable shape.

### ⚙️ Smart Preprocessing Tools
- Built-in **scaling options**: `'none'`, `'minmax'`, `'standard'`
- Manual implementation of normalization and standardization for full transparency.
- Built-in **validation checks** to prevent data mismatch or improper scaling.

### ⚡ Optimized Training Loop
- Implements **stochastic gradient descent** with up to **2.5 million iterations**.
- Supports **early stopping** based on tolerance and convergence.
- Configurable **learning rate, validation split, and tolerance levels**.
- Verbose training output includes iteration progress, current loss, and convergence messages.

### 📊 Prediction & Evaluation
- Predicts using fitted weights and bias for all supported tasks.
- Offers an **evaluation module** including:
  - **Regression Metrics:** MSE, RMSE, RMSLE
  - **Binary Metrics:** Accuracy, Precision, Recall, F1
  - **Multi-class Metrics:** Weighted Accuracy, Precision, Recall, F1
- Built-in scoring wrapper simplifies evaluation for both beginners and pros.

### 🎨 Developer-Friendly Output
- Uses **colorama** for color-coded terminal logs:
  - 🟢 Success
  - 🟡 Warning
  - 🔴 Error
- Provides rich textual feedback on training state and potential improvements.

---

## 🚀 Quickstart Guide

### 1️⃣ Install the package
```bash
pip install IntelliNeuro==0.1.0
```

### 2️⃣ Import and initialize
```python
from PerceptronX import Perceptron

model = Perceptron(
    learning_rate=0.001,
    validation_split=0.2,
    scaling='standard',
    is_scaled=False,
    tolerance=1e-6
)
```

### 3️⃣ Train the model
```python
model.fit(X_train, y_train)
```

### 4️⃣ Predict
```python
predictions = model.predict(X_test)
```

### 5️⃣ Evaluate
```python
score = model.score(X_test, y_test, metrics='accuracy')
print(f"Model Accuracy: {score}")
```

---

## 🔍 How It Works (Under the Hood)

### ⚙️ Gradient Descent Core
The perceptron updates its weights iteratively:

\[ w_{new} = w_{old} - \alpha * \nabla J(w) \]

Where:
- **\( \alpha \)** → learning rate
- **\( J(w) \)** → loss function (depends on task)

Each iteration minimizes:
- **Linear regression:** Mean Squared Error (MSE)
- **Binary classification:** Binary Cross-Entropy
- **Multi-class classification:** Categorical Cross-Entropy

### 🧩 Scaling Options
- **MinMaxScaler:** `(X - X_min) / (X_max - X_min)`
- **StandardScaler:** `(X - mean) / std`
- Optional manual toggling via `is_scaled` flag for user control.

### 🧪 Validation Split
Automatically separates validation data (based on `validation_split`), trains on the rest, and prints validation accuracy/loss after training.

### 🔔 Activation Functions
- **Sigmoid:** For binary outputs
- **Softmax:** For multi-class tasks

### 🧮 Weight Initialization
- Random normal initialization for weights.
- Zero initialization for bias.

---

## ⚠️ Important Usage Notes

- Multi-class classification is still **experimental**, designed for demonstration and learning.
- Ensure **proper data scaling** before training — incorrect scaling may slow convergence.
- Calling `predict()` before `fit()` will raise an error.
- Verbose logs can be toggled off if preferred for performance runs.

---

## 🛠 Installation Requirements

### Dependencies
```bash
pip install numpy pandas scikit-learn colorama
```

### Minimum Requirements
- Python 3.7+
- CPU: Any modern processor
- RAM: 4GB or above recommended

---

## 💡 Best Practices

✅ Always inspect your dataset with summary statistics before training.  
✅ Use standard scaling for models with large feature ranges.  
✅ Adjust learning rate carefully; small rates improve stability.  
✅ Observe tolerance-based convergence messages to avoid overfitting.  
✅ Multi-class should be used for educational visualization, not production.  

---

## 🧠 Educational Focus

**IntelliNeuro PerceptronX** isn’t just a library — it’s a *learning tool.*  
Each method is structured to demonstrate how the perceptron learns step-by-step, allowing developers to visualize how gradient descent evolves weight matrices.

This makes it a perfect resource for:
- Students exploring machine learning fundamentals.
- Educators building course materials.
- Researchers prototyping perceptron logic for custom frameworks.

---

## 🤝 Support & Contributions

**Author:** Ajay Soni  
**Email:** programmingwithcode@gmail.com  
**Repository:** https://github.com/ml-beginner-learner/IntelliNeuro

Contributions are always welcome — whether through bug reports, performance suggestions, or new features!  
Please fork the repo and submit a pull request.

---

## 📄 License

Licensed under the **MIT License** — feel free to modify, distribute, and enhance the code with proper credit.

---

## ⭐ Closing Note

If you found **IntelliNeuro PerceptronX** useful or insightful, don’t forget to leave a **⭐ on GitHub** and share it with your developer peers. Together, we build open and transparent ML tools for the next generation.
