# 🏠 House Price Prediction

## 📘 Overview
This project predicts house prices based on various features such as area, number of rooms, location, and other property characteristics.  
It uses **machine learning regression models** (like Linear Regression, Decision Tree, Random Forest, etc.) to estimate house prices accurately.

---

## 🧠 Objectives
- Understand and preprocess real estate data  
- Train multiple regression models  
- Compare model performances  
- Predict prices for new data  

---

## 🧩 Features Used
Typical features (depending on dataset):
- `area` — total square feet of the house  
- `bedrooms` — number of bedrooms  
- `bathrooms` — number of bathrooms  
- `location` — area or region  
- `price` — target variable to predict  

---

## ⚙️ Tools and Libraries
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🚀 Steps Involved
1. **Import Libraries**  
2. **Load Dataset**  
3. **Data Cleaning** (handle missing values, outliers, etc.)  
4. **Feature Selection**  
5. **Split Data** using `train_test_split()`  
6. **Train Models**
   - Linear Regression  
   - Decision Tree Regressor  
   - Random Forest Regressor  
   - KNN Regressor  
7. **Evaluate Models** using metrics like:
   - R² Score  
   - Mean Absolute Error (MAE)  
   - Mean Squared Error (MSE)  
8. **Predict Prices** for new inputs  

---

## 📊 Example Code Snippet
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = df.drop('price', axis=1)
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```

---

## 🧾 Results
The trained model gives predicted house prices and helps compare how different algorithms perform on the same dataset.

---

## 📈 Future Improvements
- Add more advanced models (XGBoost, CatBoost, etc.)  
- Use feature engineering to improve accuracy  
- Deploy the model using Streamlit or Flask  

---

## 👨‍💻 Author
**Khalith Syed**  
Student Project | Machine Learning Practice
