import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -------------------------
# Configuration
# -------------------------
DATA_FILE = "Dataset.csv"
MODEL_DIR = "models"

print("\n==============================")
print("🚀 MODEL TRAINING STARTED")
print("==============================\n")

# -------------------------
# Load Dataset
# -------------------------
if not os.path.exists(DATA_FILE):
    print("❌ ERROR: Dataset.csv not found!")
    input("\nPress ENTER to exit...")
    exit()

df = pd.read_csv(DATA_FILE)
print("📄 Dataset Loaded Successfully.\n")

# -------------------------
# Auto-detect required columns
# -------------------------
required_cols = {
    "price": None,
    "city": None,
    "cuisines": None,
    "rating": None
}

for col in df.columns:
    name = col.lower().strip()
    if "price" in name:
        required_cols["price"] = col
    elif name == "city":
        required_cols["city"] = col
    elif "cuisine" in name:
        required_cols["cuisines"] = col
    elif "aggregate rating" in name or name == "rating":
        required_cols["rating"] = col

missing = [k for k, v in required_cols.items() if v is None]
if missing:
    print(f"❌ Missing required fields in dataset: {missing}")
    input("\nPress ENTER to exit...")
    exit()

print("✔ Required columns detected.\n")

# -------------------------
# Select and prepare data
# -------------------------
df = df[[required_cols["price"], required_cols["city"], required_cols["cuisines"], required_cols["rating"]]].dropna()

print("✔ Data filtered and cleaned.\n")

# -------------------------
# Label Encoding
# -------------------------
label_encoders = {}

for col in [required_cols["price"], required_cols["city"], required_cols["cuisines"]]:
    df[col] = df[col].astype(str)
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    label_encoders[col] = encoder

print("✔ Encoding completed.\n")

# -------------------------
# Train/Test Setup
# -------------------------
X = df[[required_cols["price"], required_cols["city"], required_cols["cuisines"]]]
y = df[required_cols["rating"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# Train Multiple Models
# -------------------------
models = {
    "linear_regression.pkl": LinearRegression(),
    "decision_tree.pkl": DecisionTreeRegressor(max_depth=6),
    "random_forest.pkl": RandomForestRegressor(n_estimators=50, max_depth=8)
}

print("🚧 Training Models...")

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = round(r2_score(y_test, preds), 3)
    results[model_name] = score
    print(f"   📌 {model_name} → R² Score: {score}")

# -------------------------
# Save Models
# -------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

for model_name, model in models.items():
    with open(os.path.join(MODEL_DIR, model_name), "wb") as file:
        pickle.dump(model, file)

with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as file:
    pickle.dump(label_encoders, file)

print("\n==============================")
print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("==============================\n")

print("📦 Files saved inside 'models' folder:")
for file in models.keys():
    print(f"   ✔ {file}")
print("   ✔ label_encoders.pkl")


