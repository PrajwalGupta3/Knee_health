from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("knee_classifier.pkl")

# Label encodings
age_map = {'Young': 0, 'Young Adult': 1, 'Adult': 2, 'Older Adult': 3, 'Senior': 4}
gender_map = {'Male': 1, 'Female': 0, 'Non-binary': 2}

# Diagnosis function with medical insights
def diagnose_knee(data):
    results = []

    if data["ageGroup"] in ["Older Adult", "Senior"] and data["pressure"] > 700:
        results.append("<b>Osteoarthritis</b>: Due to age and high joint pressure, there may be cartilage degradation in the knee joint. This often leads to stiffness and pain during motion.")

    if data["temperature"] > 37.5:
        results.append("<b>Infection or Inflammation</b>: Elevated body temperature suggests possible systemic or localized inflammation. This may indicate conditions like septic arthritis or bursitis.")

    if abs(data["gyroX"]) > 0.5 or abs(data["gyroY"]) > 0.5:
        results.append("<b>Meniscus Tear</b>: Abnormal rotational movement detected. Twisting forces while bearing weight can damage the meniscus, causing pain and reduced mobility.")

    if abs(data["accelZ"] - 9.8) > 1.5:
        results.append("<b>Ligament Strain or Tear</b>: Vertical acceleration deviates significantly from normal gravity (9.8 m/s²). This suggests possible instability or trauma affecting the ACL or MCL.")

    return results

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    diagnosis = []
    if request.method == "POST":
        form = request.form
        data = {
            "ageGroup": form["ageGroup"],
            "height": float(form["height"]),
            "weight": float(form["weight"]),
            "gender": form["gender"],
            "temperature": float(form["temperature"]),
            "pressure": float(form["pressure"]),
            "gyroX": float(form["gyroX"]),
            "gyroY": float(form["gyroY"]),
            "gyroZ": float(form["gyroZ"]),
            "accelX": float(form["accelX"]),
            "accelY": float(form["accelY"]),
            "accelZ": float(form["accelZ"]),
        }

        input_vector = np.array([[ 
            age_map[data["ageGroup"]],
            data["height"],
            data["weight"],
            gender_map[data["gender"]],
            data["temperature"],
            data["pressure"],
            data["gyroX"],
            data["gyroY"],
            data["gyroZ"],
            data["accelX"],
            data["accelY"],
            data["accelZ"]
        ]])

        pred = model.predict(input_vector)[0]
        prediction = "Healthy Knee" if pred == 0 else "Potentially Unhealthy Knee"
        diagnosis = diagnose_knee(data) if pred == 1 else []

    return render_template("index.html", prediction=prediction, diagnosis=diagnosis)

if __name__ == "__main__":
    app.run(debug=True)
