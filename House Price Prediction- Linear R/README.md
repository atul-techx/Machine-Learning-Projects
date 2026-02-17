# 🏠 House Price Prediction using Linear Regression

This project implements a **Linear Regression model** to predict house prices
based on key features such as **area, number of bedrooms, and bathrooms**.
It uses a **real-world dataset (Kaggle)** and follows a complete
end-to-end Machine Learning workflow.

## 📌 Project Overview

- **Goal:** Predict house prices using Linear Regression
- **Type:** Supervised Machine Learning (Regression)
- **Dataset:** Kaggle House Price Dataset
- **Language:** Python

## 📂 Dataset Details

**Source:** Kaggle  
**Target Variable:** `price`

### Features Used:
- `area` – Size of the house (sq ft)
- `bedrooms` – Number of bedrooms
- `bathrooms` – Number of bathrooms

## 🛠️ Technologies & Libraries

- Python 🐍  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

## 📁 Project Structure
```text
House Price Prediction- Linear R/
│
├── data/
│ └── house_price.csv
│
├── notebooks/
│ └── house_price_linear_regression.ipynb
│
├── src/
│ └── linear_regression.py
│
├── Requirements.txt
└── README.md
```


## ⚙️ How to Run the Project

1️⃣ Clone the repository
```bash
git clone https://github.com/atul-techx/Machine-Learning-Projects.git
```
2️⃣ Go to project folder
```bash
cd "House Price Prediction- Linear R"
```
3️⃣ Create virtual environment (optional but recommended)
```bash
python -m venv venv
```
4️⃣ Activate virtual environment
Windows
```bash
venv\Scripts\activate
```
Mac / Linux
```bash
source venv/bin/activate
```
5️⃣ Install requirements
```bash
pip install -r Requirements.txt
```
6️⃣ Run the project
```bash
python src/linear_regression.py
```
