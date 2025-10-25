from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model
with open("weather_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoders
with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

le_precip = encoders['Precip Type']

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Collect numeric inputs
            temperature = float(request.form["temperature"])
            apparent_temperature = float(request.form["apparent_temperature"])
            humidity = float(request.form["humidity"])
            wind_speed = float(request.form["wind_speed"])
            wind_bearing = float(request.form["wind_bearing"])
            visibility = float(request.form["visibility"])
            cloud_cover = float(request.form["cloud_cover"])
            pressure = float(request.form["pressure"])

            # Collect and encode categorical input
            precip_type = request.form["precip_type"]
            precip_encoded = le_precip.transform([precip_type])[0]

            # Create feature array (9 features)
            features = np.array([[temperature, apparent_temperature, humidity,
                                  wind_speed, wind_bearing, visibility,
                                  cloud_cover, pressure, precip_encoded]])

            # Make prediction
            pred = model.predict(features)[0]
            prediction = str(pred)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
