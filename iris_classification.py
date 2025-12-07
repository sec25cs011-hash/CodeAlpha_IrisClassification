import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv("Iris.csv")

# Remove ID column if present
df = df.drop(columns=["Id"], errors="ignore")

# Encode Output
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Split X and y
X = df.drop("Species", axis=1)
y = df["Species"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("🔥 Model Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\n📌 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# Pairplot (Beautiful Visualization)
sns.pairplot(df, hue="Species", diag_kind="kde",
             palette="bright")
plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.show()

# Scatter Plot
plt.figure(figsize=(7, 5))
plt.scatter(df["PetalLengthCm"], df["PetalWidthCm"], c=df["Species"], cmap="viridis")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Petal Length vs Width")
plt.show()

# Predict with user sample (just to show output)
sample = [[5.2, 3.5, 1.5, 0.2]]  # Custom test input
sample_pred = model.predict(sample)
print("\n🎉 Example Prediction:", le.inverse_transform(sample_pred)[0])
