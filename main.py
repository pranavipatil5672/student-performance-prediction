import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Marks': [20, 30, 35, 50, 55, 65, 75, 85]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Marks']

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[9]])

print("Predicted Marks:", prediction[0])