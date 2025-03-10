import flask
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import base64
import requests
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from folium.plugins import HeatMap
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from models import db, DonationRequest, Donation
import threading
import time

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///donations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database with the app
db.init_app(app)

# Load LSTM model for flood prediction
model = load_model("flood_prediction_lstm.h5")

# Visual Crossing API Key
API_KEY = "QEEMNFQDYB48AV3V3SVN2BX3W"

# List of locations for flood prediction
LOCATIONS = [
    (9.31575, 76.61513), (9.3258216, 76.5759543), (9.3267379, 76.6869087),
    (10.10764, 76.35158), (10.311879, 76.331978), (10.770388, 76.37706),
    (9.2034089, 76.7121783), (10.69967, 76.7471), (10.9608223, 76.2338893),
    (12.2604, 75.10846)
]

def get_weather_data(lat, lon):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}?unitGroup=metric&key={API_KEY}&contentType=json"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    
    data = response.json().get('days', [])
    if not data:
        return None
    
    today = data[0]  # Get today's data
    
    features = np.array([
        lat,  # latitude
        lon,  # longitude
        datetime.strptime(today['datetime'], "%Y-%m-%d").timestamp(),  # Convert date to timestamp
        today.get('temp', 0),
        today.get('dew', 0),
        today.get('humidity', 0),
        today.get('precip', 0),
        today.get('precipprob', 0),
        today.get('precipcover', 0),
        today.get('windgust', 0),
        today.get('windspeed', 0),
        today.get('winddir', 0),
        today.get('sealevelpressure', 0),
        today.get('cloudcover', 0),
        today.get('solarradiation', 0),
        today.get('solarenergy', 0),
        today.get('moonphase', 0)
    ]).reshape(1, 1, -1)  # Reshape for LSTM input
    return features

# Predict flood risk using LSTM model
def predict_flood(features):
    prediction = model.predict(features)
    return 1 if prediction[0][0] > 0.5 else 0  # Threshold of 0.5

# Route to generate heatmap data
@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    heat_data = []
    for lat, lon in LOCATIONS:
        weather_data = get_weather_data(lat, lon)
        if weather_data is not None:
            flood_risk = predict_flood(weather_data)
            intensity = 1 if flood_risk == 1 else 0.2  # Higher intensity for flood risk
            heat_data.append([lat, lon, intensity])
    return jsonify(heat_data)

@app.route('/update_heatmap', methods=['POST'])
def update_heatmap():
    locations = [
        (9.31575, 76.61513), (9.3258216, 76.5759543), (9.3267379, 76.6869087),
        (10.10764, 76.35158), (10.311879, 76.331978), (10.770388, 76.37706),
        (9.2034089, 76.7121783), (10.69967, 76.7471), (10.9608223, 76.2338893),
        (12.2604, 75.10846)
    ]
    
    heatmap_data = []

    for lat, lon in locations:
        weather_data = get_weather_data(lat, lon)  # Fixed: Changed get_data to get_weather_data
        flood_risk = predict_flood(weather_data)  # Get flood prediction (0 or 1)

        # Assign intensity for heatmap (higher for flood-prone areas)
        intensity = 0.8 if flood_risk == 1 else 0.2
        heatmap_data.append({"lat": lat, "lon": lon, "intensity": intensity})

    return jsonify(heatmap_data)

data = [{'name':'Delhi', "sel": "selected"}, {'name':'Mumbai', "sel": ""}, {'name':'Kolkata', "sel": ""}, {'name':'Bangalore', "sel": ""}, {'name':'Chennai', "sel": ""}]
months = [{"name":"May", "sel": ""}, {"name":"June", "sel": ""}, {"name":"July", "sel": "selected"}]
cities = [{'name':'Delhi', "sel": "selected"}, {'name':'Mumbai', "sel": ""}, {'name':'Kolkata', "sel": ""}, {'name':'Bangalore', "sel": ""}, {'name':'Chennai', "sel": ""}, {'name':'New York', "sel": ""}, {'name':'Los Angeles', "sel": ""}, {'name':'London', "sel": ""}, {'name':'Paris', "sel": ""}, {'name':'Sydney', "sel": ""}, {'name':'Beijing', "sel": ""}]

@app.route("/")
@app.route('/index.html')
def index() -> str:
    """Base page."""
    return flask.render_template("index.html")

@app.route('/plots.html')
def plots():
    return render_template('plots.html')

@app.route('/heatmaps.html')
def heatmaps():
    return render_template('heatmaps.html')

@app.route('/chart.html')
def chart():
    return render_template('chart.html')

@app.route('/satellite.html')
def satellite():
    direc = "satellite_images/Delhi_July.png"
    with open(direc, "rb") as image_file:
        image = base64.b64encode(image_file.read())
    image = image.decode('utf-8')
    return render_template('satellite.html', data=data, image_file=image, months=months, text="Delhi in January 2024")

@app.route('/satellite.html', methods=['GET', 'POST'])
def satelliteimages():
    place = request.form.get('place')
    date = request.form.get('date')
    data = [{'name':'Delhi', "sel": ""}, {'name':'Mumbai', "sel": ""}, {'name':'Kolkata', "sel": ""}, {'name':'Bangalore', "sel": ""}, {'name':'Chennai', "sel": ""}]
    months = [{"name":"May", "sel": ""}, {"name":"June", "sel": ""}, {"name":"July", "sel": ""}]
    for item in data:
        if item["name"] == place:
            item["sel"] = "selected"
    
    for item in months:
        if item["name"] == date:
            item["sel"] = "selected"

    text = place + " in " + date + " 2024"

    direc = "satellite_images/{}_{}.png".format(place, date)
    with open(direc, "rb") as image_file:
        image = base64.b64encode(image_file.read())
    image = image.decode('utf-8')
    return render_template('satellite.html', data=data, image_file=image, months=months, text=text)

@app.route('/predicts.html')
def predicts():
    return render_template('predicts.html', cities=cities, cityname="Information about the city")

@app.route('/predicts.html', methods=["GET", "POST"])
def get_predicts():
    cities = [{'name': 'Chengannur', "sel": ""}, {'name': 'Pandanad', "sel": ""}, {'name': 'Aranmula', "sel": ""},
              {'name': 'Aluva', "sel": ""}, {'name': 'Chalakudy', "sel": ""}, {'name': 'Kuttanad', "sel": ""},
              {'name': 'Pandalam', "sel": ""}, {'name': 'Chittur', "sel": ""}, {'name': 'Perinthalmanna', "sel": ""},
              {'name': 'Nileshwar', "sel": ""}]

    temp = maxt = wspd = cloudcover = percip = humidity = pred = None  # Default values

    if request.method == "POST":
        try:
            cityname = request.form["city"]
            for item in cities:
                if item['name'] == cityname:
                    item['sel'] = 'selected'

            # Get latitude and longitude from geocoding API
            URL = "https://geocode.search.hereapi.com/v1/geocode"
            api_key = 'pPFSt0miNxLZJY6_Zs-h-nB9W1XxxJG6s3wat1L37r8'
            PARAMS = {'apikey': api_key, 'q': cityname}
            response = requests.get(url=URL, params=PARAMS)
            data = response.json()
            lat, lon = data['items'][0]['position']['lat'], data['items'][0]['position']['lng']

            # Get weather data
            weather_data = get_weather_data(lat, lon)

            if weather_data is not None:
                # Extract features
                temp = round(float(weather_data[0, 0, 3]), 2)  # Temperature
                maxt = round(float(weather_data[0, 0, 3]) + 2, 2)  # Estimated Max Temperature
                wspd = round(float(weather_data[0, 0, 10]), 2)  # Wind Speed
                cloudcover = round(float(weather_data[0, 0, 13]), 2)  # Cloud Cover
                percip = round(float(weather_data[0, 0, 6]), 2)  # Precipitation
                humidity = round(float(weather_data[0, 0, 5]), 2)  # Humidity

                # Flood Prediction
                flood_risk = predict_flood(weather_data)
                pred = "Safe" if flood_risk == 1 else "Unsafe"

            return render_template(
                'predicts.html',
                cityname=f"Information about {cityname}",
                cities=cities,
                temp=temp,
                maxt=maxt,
                wspd=wspd,
                cloudcover=cloudcover,
                percip=percip,
                humidity=humidity,
                pred=pred
            )
        except:
            return render_template('predicts.html', cities=cities, cityname="Oops, we weren't able to retrieve data for that city.")

    return render_template('predicts.html', cities=cities, cityname="Information about the city")


# Route to fetch all requests
@app.route('/get_requests', methods=['GET'])
def get_requests():
    requests = DonationRequest.query.all()
    requests_data = [{
        'id': req.id,
        'name': req.name,
        'location': req.location,
        'help_type': req.help_type,
        'status': req.status,
        'timestamp': req.timestamp.isoformat() if req.timestamp else None
    } for req in requests]
    return jsonify(requests_data)

# Route for donation page
@app.route('/donation.html')
def donation():
    return render_template('donation.html')

# Route to submit a new donation request
@app.route('/submit_request', methods=['POST'])
def submit_request():
    name = request.form['name']
    location = request.form['location']
    help_type = request.form['help_type']
    new_request = DonationRequest(name=name, location=location, help_type=help_type, timestamp=datetime.now())
    db.session.add(new_request)
    db.session.commit()
    return redirect(url_for('donation'))  # Redirect to the new donation route

# Route to handle donations
@app.route('/donate', methods=['POST'])
def donate():
    request_id = request.form['request_id']
    donor_name = request.form['donor_name']
    donation_type = request.form['donation_type']

    # Create a new donation entry
    new_donation = Donation(donor_name=donor_name, donation_type=donation_type, status="Completed")
    db.session.add(new_donation)

    # Update the status of the donation request
    request_entry = DonationRequest.query.get(request_id)
    if request_entry:
        request_entry.status = "Fulfilled"
        request_entry.donated_item = donation_type  # Add a new field to store the donated item
        request_entry.timestamp = datetime.now()  # Update timestamp when fulfilled

    db.session.commit()

    # Return the updated request data
    updated_request = {
        'id': request_entry.id,
        'name': request_entry.name,
        'location': request_entry.location,
        'help_type': request_entry.help_type,
        'status': request_entry.status,
        'donated_item': request_entry.donated_item,  # Include the donated item
        'timestamp': request_entry.timestamp.isoformat() if request_entry.timestamp else None
    }
    return jsonify({"success": True, "request": updated_request})
# Background task to delete fulfilled requests older than a minute
def delete_old_fulfilled_requests():
    with app.app_context():
        while True:
            time.sleep(60)  # Check every minute
            now = datetime.now()
            fulfilled_requests = DonationRequest.query.filter_by(status="Fulfilled").all()
            for req in fulfilled_requests:
                if now - req.timestamp > timedelta(minutes=1):
                    db.session.delete(req)
            db.session.commit()

# Start the background thread
threading.Thread(target=delete_old_fulfilled_requests, daemon=True).start()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)