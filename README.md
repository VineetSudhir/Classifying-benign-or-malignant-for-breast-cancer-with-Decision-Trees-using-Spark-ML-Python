# Classifying Breast Cancer Tumors using Decision Trees (PySpark ML)

This project builds a classification pipeline using **PySpark MLlib** to detect whether a tumor is **benign or malignant** based on the **Wisconsin Breast Cancer dataset**. It includes **feature engineering**, **model training**, **hyperparameter tuning**, and **decision tree visualization**.

---

## 🧠 Project Overview

- **Framework:** PySpark MLlib  
- **Model:** Decision Tree Classifier  
- **Goal:** Classify tumors as benign or malignant  
- **Dataset:** Breast Cancer Wisconsin (Original) — [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original))

---

## 🔧 Workflow Summary

1. **Data Preprocessing**
   - Loaded dataset using Spark DataFrame API
   - Handled missing and categorical values
   - Indexed labels and one-hot encoded categorical features

2. **Feature Engineering**
   - Used `VectorAssembler` to prepare input features
   - Extracted relevant features like `clump_thickness`, `bare_nuclei`, etc.

3. **Model Training & Evaluation**
   - Trained a **Decision Tree Classifier** on training data
   - Evaluated on test set using **F1 score**
   - Achieved F1 score up to **0.978** on unseen data

4. **Pipeline & Automation**
   - Built a **full ML pipeline** with label encoding, feature transformation, and model fitting
   - Implemented automated **hyperparameter tuning** for `maxDepth` and `minInstancesPerNode`

5. **Visualization**
   - Parsed trained decision trees using custom tools
   - Plotted tree structures as interactive HTML visualizations

---

## 📈 Key Results

| Step                         | F1 Score |
|-----------------------------|----------|
| Initial Model (no tuning)   | ~0.978   |
| Final Tuned Model           | ~0.973   |

> ✅ **Best Parameters:** `maxDepth = 6`, `minInstancesPerNode = 2`  
> 📊 **Visualization:** Interactive tree visual saved as HTML

---

## 🛠️ Technologies Used

- PySpark MLlib (DecisionTreeClassifier, VectorAssembler, StringIndexer)
- Pandas for evaluation tracking
- Custom visualization via `decision_tree_plot` package
- Apache Spark Session for scalable execution

---

## 📂 Output

- Evaluation results saved as text files via Spark RDD
- Final model visualized and saved to: `/Lab7DT/bestDTtree2B.html`

---

## 🚀 How to Run

> Make sure Apache Spark is installed and your environment is properly configured.

```bash
spark-submit breast_cancer_decision_tree.py
