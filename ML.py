import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold,cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


data=pd.read_csv("AI_SocialMedia_Student_Dataset.csv")
print(data)

known = data.notnull()
missing = data.isnull()
print(known)
print(missing)

encode = OrdinalEncoder()
category = [["low"], ["medium"], ["high"]]
encoded_data = encode.fit_transform(category)
print(encoded_data)

num_data = data[[
    "Age",
    "Daily_Social_Media_Hours",
    "Daily_AI_Tool_Usage_Hours",
    "Sleep_Hours",
    "Physical_Activity_Hours",
    "Mental_Health_Score",
    "Physical_Health_Score"
]]

Q1 = num_data.quantile(0.25)
Q3 = num_data.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("lower limit:", lower)
print("upper limit:", upper)
print((num_data < lower) | (num_data > upper))

x = data[[
    "Age",
    "Daily_Social_Media_Hours",
    "Daily_AI_Tool_Usage_Hours",
    "Sleep_Hours",
    "Physical_Activity_Hours",
    "Mental_Health_Score",
    "Physical_Health_Score"
]]

y = data["Education_Level"]
y = y.astype("category").cat.codes

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
Kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

R2_scores = cross_val_score(
    model,
    x,
    y,
    cv=Kf,
    scoring="r2"
)

accuracy_scores = cross_val_score(
    model,
    x,
    y,
    cv=Kf,
    scoring="accuracy"
)

print("R2 score for each fold:")

for i, Score in enumerate(R2_scores, 1):
    print(f"Fold {i}: {Score:.4f}")

print("\nAverage R2 score:",np.mean(R2_scores))
print("\nStandard Deviation:",np.std(R2_scores))

print("Accuracy for each fold:")

for i, score in enumerate(accuracy_scores, 1):
    print(f"Fold {i}: {score:.4f}")

print("\nAverage Accuracy:", np.mean(accuracy_scores))
print("Accuracy Standard Deviation:", np.std(accuracy_scores))


y_pred = cross_val_predict(
    model,
    x,
    y,
    cv=Kf
)

accuracy = accuracy_score(y, y_pred)

print("\nOverall Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y,
    y_pred,
    target_names=["Low", "Medium", "High"]
))

cm = confusion_matrix(y, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    yticklabels=["High","Medium","Low"],
    xticklabels=["Low","Medium","High"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

plt.figure(figsize=(7, 4))

plt.bar(
    range(1, 6),
    accuracy_scores,
    width=0.6
)

plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Accuracy for Each Fold")

plt.xticks(range(1, 6))
plt.ylim(0,1)
plt.tight_layout()
plt.show()

pickle.dump(
    model,
    open("models.pkl","wb")
)
print("Model Saved")


