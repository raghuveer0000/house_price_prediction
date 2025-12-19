import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib


data = pd.read_csv("Housing.csv")

X = data.drop("price", axis=1)
y = data["price"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(include=["int64"]).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)


model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model.fit(X_train, y_train)


joblib.dump(model, "house_price_model.pkl")
print(" Model trained and saved as house_price_model.pkl")



print("\n--- Enter House Details for Prediction ---")

area = int(input("Enter area (sq. ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
stories = int(input("Enter number of stories: "))
mainroad = input("Is it on main road? (yes/no): ").strip().lower()
guestroom = input("Guestroom available? (yes/no): ").strip().lower()
basement = input("Basement available? (yes/no): ").strip().lower()
hotwaterheating = input("Hot water heating? (yes/no): ").strip().lower()
airconditioning = input("Air conditioning? (yes/no): ").strip().lower()
parking = int(input("Number of parking spaces: "))
prefarea = input("Preferred area? (yes/no): ").strip().lower()
furnishingstatus = input("Furnishing status (furnished/semi-furnished/unfurnished): ").strip().lower()


new_data = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus
}])

e
predicted_price = model.predict(new_data)[0]
print(f"\n Predicted House Price: ₹{predicted_price:,.0f}")
