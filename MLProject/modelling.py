import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv('diabetes_clean.csv')

# =========================
# SPLIT DATA
# =========================
X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# ENABLE AUTOLOG
# =========================
mlflow.sklearn.autolog()

# =========================
# TRAINING MODEL
# =========================
with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # =========================
    # PREDIKSI
    # =========================
    y_pred = model.predict(X_test)

    # =========================
    # HITUNG AKURASI
    # =========================
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

