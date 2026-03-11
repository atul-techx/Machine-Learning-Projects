import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r"C:\Users\pintu\OneDrive\Desktop\classpractice\class_assignment\cleaned.csv")
print(df.columns)
print(df.head())
df_encoded = pd.get_dummies(df['Type_of_collision'], drop_first=True)
X = df_encoded
y = df['Accident_severity']
# Train model
model = LinearRegression()
model.fit(X, y)
# Prediction
y_pred = model.predict(X)
# Plot
plt.figure(figsize=(8,5))
plt.scatter(range(len(y)), y, label="Actual Severity")
plt.plot(range(len(y)), y_pred, color='red', label="Predicted Line")
plt.xlabel("Samples")
plt.ylabel("Accident Severity")
plt.title("Accident Severity Prediction")
plt.legend()
plt.show()