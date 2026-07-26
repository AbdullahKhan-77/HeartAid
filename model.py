import pandas as pd
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("dataset/heart.csv")
resting_bp_median = df['RestingBP'].median()
df.loc[df['RestingBP']==0,'RestingBP'] =resting_bp_median    
chol_medians_by_group = df.groupby('HeartDisease')['Cholesterol'].transform('median')
df.loc[df['Cholesterol']==0,'Cholesterol'] = chol_medians_by_group[df['Cholesterol']==0]

df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)

# Separate X (features) and y (target)
X = df_encoded.drop(columns=['HeartDisease'])
y = df_encoded['HeartDisease']

# Train/test split stratified on y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)  

logreg = LogisticRegression()
logreg.fit(X_train_scaled,y_train)

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train, y_train) 


gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train) 

# Predictions
logreg_preds = logreg.predict(X_test_scaled)
rf_preds = rf.predict(X_test)        

cm_logreg = confusion_matrix(y_test, logreg_preds)
cm_rf = confusion_matrix(y_test, rf_preds)

gb_preds = gb.predict(X_test)
cm_gb=confusion_matrix(y_test, gb_preds)
# Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_preds))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("Gradient Boosting Classifier Accuracy",accuracy_score(y_test, gb_preds))

# Confusion matrices
print("Logistic Regression Confusion Matrix:")
print(cm_logreg)

print("Random Forest Confusion Matrix:")
print(cm_rf)

print("Gradient Boosting Classifier Confusion Matrix")
print(cm_gb)

os.makedirs("models", exist_ok=True)
save_path = os.path.join("models", "heart_disease_model.joblib")

joblib.dump(gb, save_path)
print(f"Model saved to {save_path}")

loaded_model = joblib.load(save_path)
sample = X_test.iloc[[0]]
prediction = loaded_model.predict(sample)[0]
actual = y_test.iloc[0]
print(f"Predicted: {prediction} | Actual: {actual}")