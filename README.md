# 🧠 Automated Technical Debt Analyzer

An interactive **AST-based software maintainability analysis system** that predicts
**technical debt severity** in Python source code using **Machine Learning** and
**Graph Neural Networks (GNNs)**.

The application is deployed using **Streamlit** and allows users to paste
Python code (including GitHub-sourced or AI-generated code) to assess
maintainability risk in real time.

---

## 🚀 Features

- 🔍 **AST-based feature extraction** from Python source code
- 🌳 **Random Forest Regression**
  - Fast and accurate in low-data settings
- 🧠 **Graph Neural Network (GNN)**
  - Structural analysis of Abstract Syntax Trees
  - Experimental and research-oriented
- 🖥️ **Interactive Web Interface** using Streamlit
- 📊 **Continuous Technical Debt Score** (Regression-based)

---

## 🧪 Models Used

### Random Forest Regressor
- Operates on aggregated AST metrics such as:
  - Number of nodes
  - Loops
  - Conditionals
  - Function definitions
- Performs well with limited training data

### Graph Neural Network (GNN)
- Represents source code as an **AST graph**
- Learns structural patterns via message passing
- Useful for exploratory research into graph-based code analysis

---

## 🧠 How It Works

1. User pastes Python source code into the web interface
2. The system parses the code into an **Abstract Syntax Tree (AST)**
3. Structural features are extracted from the AST
4. Selected model predicts a **technical debt severity score**
5. Results are displayed instantly in the browser

---

## 📦 Technologies Used

- Python
- Streamlit
- Scikit-learn
- PyTorch
- PyTorch Geometric
- NetworkX

---

## 🎓 Academic Context

This project was developed as part of an **MSc in Data Analytics** and focuses on:

- Machine Learning
- Software Quality & Maintainability
- Technical Debt Prediction
- Graph-based Representations of Code

---

## ⚠️ Notes

- GNN predictions are **experimental** and depend on learned structural patterns
- Random Forest models provide more stable results in low-data scenarios
- Model comparison highlights how different ML paradigms interpret code quality

---

## 📜 License

This project is intended for **academic and research purposes**.
