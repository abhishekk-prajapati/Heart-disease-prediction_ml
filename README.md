# ❤️ Heart Disease Predictor

![Python Feature](https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

An end-to-end Machine Learning web application designed to predict the likelihood of heart disease in patients based on 13 clinical attributes.

**Unique Project Highlight:** The core machine learning algorithms (Logistic Regression, K-Nearest Neighbors, and Random Forest) used in the interactive Streamlit dashboard are built **entirely from scratch using only NumPy**, without relying on popular frameworks like `scikit-learn`. This demonstrates a deep, fundamental understanding of the underlying mathematics (gradient descent, Euclidean distance, information gain, bootstrap sampling).

---

## 🚀 Features

- **Interactive UI**: A fully responsive web interface built with Streamlit allowing real-time clinical data input.
- **Algorithms from Scratch**: 
  - **Logistic Regression**: Implementation includes forward pass, binary cross-entropy loss, and backpropagation via gradient descent.
  - **K-Nearest Neighbors (KNN)**: Custom Euclidean distance calculator and majority voting mechanism.
  - **Random Forest**: Comprehensive CART-style decision tree from scratch, featuring information gain validation, feature bagging, and bootstrap sampling (ensemble learning).
- **Data Pre-processing**: Custom implementations for standardisation (Z-score normalisation), train/test splitting, and classification metrics (Accuracy, Precision, Recall, F1-Score).
- **Real-time Evaluation**: Instantly trains the chosen algorithm on the Cleveland dataset and generates a probability score alongside a confusion matrix.

---

## 📊 Dataset

This project utilizes the processed **Cleveland Clinic Heart Disease Dataset** from the UCI Machine Learning Repository. It contains 303 patient records with 13 continuous/categorical features and 1 binary target.

### Clinical Attributes Used:
1. `age`: Age in years
2. `sex`: 1 = male; 0 = female
3. `cp`: Chest pain type (4 values)
4. `trestbps`: Resting blood pressure (in mm Hg)
5. `chol`: Serum cholesterol in mg/dl
6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. `restecg`: Resting electrocardiographic results (values 0,1,2)
8. `thalach`: Maximum heart rate achieved
9. `exang`: Exercise-induced angina (1 = yes; 0 = no)
10. `oldpeak`: ST depression induced by exercise relative to rest
11. `slope`: The slope of the peak exercise ST segment (values 1,2,3)
12. `ca`: Number of major vessels (0-3) colored by fluoroscopy
13. `thal`: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)

---

## 🛠️ Installation & Setup

To run this project locally on your machine, follow these steps:

### Prerequisites Ensure you have Python installed.

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/heart-disease-predictor.git
cd heart-disease-predictor
```

**2. Install required dependencies**
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```
*(If you do not have a requirements.txt, you can install the packages directly:)*
```bash
pip install streamlit pandas numpy scikit-learn
```

**3. Run the application**
```bash
streamlit run app.py
```

The application will launch on your default web browser at `http://localhost:8501`.

---

## 🧠 Model Architecture & Methodology

While testing tools like Scikit-Learn (`model_methods.py`) are present for benchmarking, the primary application (`app.py`) operates strictly using custom NumPy logic.

| Algorithm | How it was built (Scratch Implementation) |
| --- | --- |
| **Logistic Regression** | Linear combination mapped through a `_sigmoid` function. Uses Binary Cross-Entropy loss and iteratively updates weights and bias via Gradient Descent (α=0.1, 1000 iter). |
| **K-Nearest Neighbours** | Lazy evaluation logic. Computes exact Euclidean distance between the input vector and all points in `X_train`. Finds indices of `K` smallest distances and determines the class via majority vote. |
| **Decision Tree (CART)** | Recursive partitioning. At every node, it calculates `_entropy` and tests random subsets of features/thresholds to find the highest `_information_gain`. |
| **Random Forest** | Employs Bootstrap Sampling to create sub-datasets. Trains multiple custom Decision Trees, applying Feature Bagging (√(n) features) to de-correlate errors, executing a majority aggregate vote for the final prediction. |

---

## 📁 Repository Structure

```
├── app.py                  # Main Streamlit application & Scratch ML classes
├── model_methods.py        # Helper benchmarking functions using Scikit-Learn tools
├── heart_data.txt          # The Cleveland clinic dataset used for training
└── dataset.txt             # Dataset origin notes and source URLs
```

---

## 👤 Author 

**[Your Name]** 
* Data Science Enthusiast | Machine Learning Engineer
* [LinkedIn](https://linkedin.com/in/yourprofile) | [Portfolio](https://yourwebsite.com) | [Email](mailto:youremail@example.com)

If you found this project helpful or interesting, please consider giving it a ⭐!
