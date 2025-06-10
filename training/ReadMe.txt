# Knee Health Classification using Machine Learning

This project is a demonstration of supervised learning applied to knee health classification. It predicts whether a knee is healthy or potentially unhealthy based on sensor and demographic data, and provides a medically sound hypothesis of what condition might be present.

---
Dataset Overview

The model was trained on 1,000 synthetically generated entries. Each entry includes:

* Age Group (categorical)
* Height (cm), Weight (kg)
* Gender (categorical)
* Body Temperature (°C)
* Joint Pressure (N)
* Gyroscope Readings (X, Y, Z)
* Accelerometer Readings (X, Y, Z)
* Target Class: Healthy or Potentially Unhealthy Knee

We use this dataset because it's structured, clean, and simulates real-world sensor data typically captured during knee assessments.

---

### RandomForestClassifier (Used Here)

For tabular data, **Random Forest** outperforms SVM in many practical situations. It handles categorical variables better and can model non-linear decision boundaries well.

If SVM were used here, we’d need careful tuning and normalization due to its sensitivity to feature scales.

---

## How the Code Works (With Analogies)

### 1. Load and Preprocess Data

```python
train_df = pd.read_csv("dataset.txt")
test_df = pd.read_csv("test.txt")
```

*Analogy*: Think of loading a recipe book (train) and a taste-test sheet (test). We're preparing to learn from the recipes and see how new dishes compare.

### 2. Combine for Consistent Label Encoding

```python
combined = pd.concat([train_df, test_df])
```

*Analogy*: Like labeling ingredients in both books with the same names before sorting them.

### 3. Encode Categorical Data

```python
for col in ["Age Group", "Gender", "Target Class"]:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
```

*Analogy*: Replacing words like "Young" with numbers, like making a color code for quick reading.

### 4. Split Again

```python
train_df = combined.iloc[:len(train_df)]
test_df = combined.iloc[len(train_df):]
```

*Analogy*: Putting recipes and tests back in their original places after updating labels.

### 5. Separate Inputs and Outputs

```python
X_train = train_df.drop(columns=["ID", "Target Class"])
y_train = train_df["Target Class"]
```

*Analogy*: Ingredients go in one bowl, the final dish name in another.

### 6. Train the Model

```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

*Analogy*: Teaching a chef by showing many examples of good vs bad knees.

### 7. Test the Model

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

*Analogy*: Letting the chef taste unknown dishes and seeing how well they guess right.

### 8. Save Model

```python
joblib.dump(model, "knee_classifier.pkl")
```

*Analogy*: Saving the chef's memory so we can reuse him later.

---

## 🔮 Making Predictions

We encode new user data into a format the model understands, just like standardizing how you present ingredients to a chef.

### Input:

* Age Group
* Height/Weight
* Gender
* Temperature
* Pressure
* Gyroscope and Accelerometer readings

### Output:

* "Healthy Knee" or "Potentially Unhealthy Knee"

---

## 🏥 Medical Reasoning

When a knee is predicted as **Potentially Unhealthy**, we add interpretability:

```python
def diagnose_knee(input_data):
    if older + high pressure -> Osteoarthritis
    if high temperature -> Infection or inflammation
    if abnormal rotation -> Meniscus tear
    if vertical acceleration weird -> Ligament strain
```

These simple rules help explain the model's thinking and offer helpful next steps.

---

## ✅ Example Output

```
Prediction: Potentially Unhealthy Knee
Possible Conditions:
- Possible Osteoarthritis due to age and high joint pressure.
- Possible inflammation or infection due to elevated body temperature.
- Possible Meniscus Tear from abnormal rotation.
```

---

## 📌 Next Steps

* Integrate this into a web or mobile app for physiotherapists
* Replace synthetic data with real-world sensor datasets
* Add visualizations for motion patterns

---

## 🤝 Contributing

Pull requests are welcome! Help make this tool smarter and more user-friendly.

---

## 🧑‍🔬 Author

This project is created as part of a deep learning exploration into medical diagnostics using non-image data. It demonstrates practical understanding of transfer learning principles, supervised learning, and model interpretability.
