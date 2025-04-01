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
API_KEY = "QDJ48TMKR5K6NGW7W99UBX7XH"
#backup1 = "BWGMFZVJGFRMYPG92N9MNYJ9G"
#backup2 = "QEEMNFQDYB48AV3V3SVN2BX3W"

# List of locations for flood prediction
LOCATIONS = [
    (11.75, 75.5667), (11.987, 75.45), (12.5, 74.99),
    (11.987, 75.376), (11.9833, 75.6833), (11.8333, 75.5667),
    (12.05, 75.35), (11.93, 75.57), (12.093, 75.202),
    (11.75, 75.5), (11.803, 76.014), (11.6667, 76.2667),
    (11.6086, 76.0827), (11.525, 75.693), (11.441, 75.932),
    (11.1793, 75.8414), (11.1796, 75.8414), (11.3046, 75.9841),
    (11.0427, 75.9221), (11.4426, 75.695), (11.608, 75.5917),
    (9.4519851, 76.5367065), (9.591564, 76.5221599), (9.2780698, 76.4424457),
    (9.3285603, 76.6165824), (9.238487, 76.531479), (9.498067, 76.338844),
    (9.668878, 76.339769), (9.183333, 76.5), (8.4065974, 77.0932133),
    (8.695034, 76.817879), (8.603333, 77.002777), (9.0066825, 76.7779672),
    (9.3816, 76.57489), (10.1475609, 76.2289395), (9.2034089, 76.7121783),
    (9.26667, 76.78333), (9.2, 76.76), (8.7333, 76.7167),
    (9.054736, 76.535358), (9.5053209, 76.7770683), (10.65, 76.0667),
    (10.3428, 76.211), (10.3, 76.3317), (9.6874, 76.78),
    (9.6686, 76.557), (10.048, 76.3083), (9.751, 77.116),
    (9.8333, 76.5833), (10.196, 76.386), (9.939, 76.3183),
    (9.9796, 76.5738), (10.1076, 76.3452), (10.1411, 76.2296),
    (10.0615, 76.6297), (10.1115, 76.4797), (9.8667, 76.5),
    (10.067, 76.292), (9.892, 76.7184), (10.028, 76.3264),
    (9.7481, 76.3964)
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
        (11.75, 75.5667), (11.987, 75.45), (12.5, 74.99),
        (11.987, 75.376), (11.9833, 75.6833), (11.8333, 75.5667),
        (12.05, 75.35), (11.93, 75.57), (12.093, 75.202),
        (11.75, 75.5), (11.803, 76.014), (11.6667, 76.2667),
        (11.6086, 76.0827), (11.525, 75.693), (11.441, 75.932),
        (11.1793, 75.8414), (11.1796, 75.8414), (11.3046, 75.9841),
        (11.0427, 75.9221), (11.4426, 75.695), (11.608, 75.5917),
        (9.4519851, 76.5367065), (9.591564, 76.5221599), (9.2780698, 76.4424457),
        (9.3285603, 76.6165824), (9.238487, 76.531479), (9.498067, 76.338844),
        (9.668878, 76.339769), (9.183333, 76.5), (8.4065974, 77.0932133),
        (8.695034, 76.817879), (8.603333, 77.002777), (9.0066825, 76.7779672),
        (9.3816, 76.57489), (10.1475609, 76.2289395), (9.2034089, 76.7121783),
        (9.26667, 76.78333), (9.2, 76.76), (8.7333, 76.7167),
        (9.054736, 76.535358), (9.5053209, 76.7770683), (10.65, 76.0667),
        (10.3428, 76.211), (10.3, 76.3317), (9.6874, 76.78),
        (9.6686, 76.557), (10.048, 76.3083), (9.751, 77.116),
        (9.8333, 76.5833), (10.196, 76.386), (9.939, 76.3183),
        (9.9796, 76.5738), (10.1076, 76.3452), (10.1411, 76.2296),
        (10.0615, 76.6297), (10.1115, 76.4797), (9.8667, 76.5),
        (10.067, 76.292), (9.892, 76.7184), (10.028, 76.3264),
        (9.7481, 76.3964)
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
    # List of 76 locations
    cities = [{'name': 'Neyyattinkara', "sel": ""}, {'name': 'Nedumangad', "sel": ""}, {'name': 'Attingal', "sel": ""}, 
              {'name': 'Varkala', "sel": ""}, {'name': 'Punalur', "sel": ""}, {'name': 'Karunagappally', "sel": ""}, 
              {'name': 'Paravur', "sel": ""}, {'name': 'Kottarakkara', "sel": ""}, {'name': 'Thiruvalla', "sel": ""}, 
              {'name': 'Pathanamthitta', "sel": ""}, {'name': 'Adoor', "sel": ""}, {'name': 'Pandalam', "sel": ""}, 
              {'name': 'Alappuzha', "sel": ""}, {'name': 'Kayamkulam', "sel": ""}, {'name': 'Cherthala', "sel": ""}, 
              {'name': 'Mavelikara', "sel": ""}, {'name': 'Chengannur', "sel": ""}, {'name': 'Haripad', "sel": ""}, 
              {'name': 'Kottayam', "sel": ""}, {'name': 'Changanacherry', "sel": ""}, {'name': 'Pala', "sel": ""}, 
              {'name': 'Pattambi', "sel": ""}, {'name': 'Shoranur', "sel": ""}, {'name': 'Kodungallur', "sel": ""}, 
              {'name': 'Chittur', "sel": ""}, {'name': 'Cherpulassery', "sel": ""}, {'name': 'Chavakkad', "sel": ""}, 
              {'name': 'Guruvayoor', "sel": ""}, {'name': 'Mannarkkad', "sel": ""}, {'name': 'Ottapalam', "sel": ""}, 
              {'name': 'Wadakkanchery', "sel": ""}, {'name': 'Palakkad', "sel": ""}, {'name': 'Tirurangadi', "sel": ""}, 
              {'name': 'Vatakara', "sel": ""}, {'name': 'Koyilandy', "sel": ""}, {'name': 'Mukkam', "sel": ""}, 
              {'name': 'Ramanattukara', "sel": ""}, {'name': 'Feroke', "sel": ""}, {'name': 'Payyoli', "sel": ""}, 
              {'name': 'Koduvally', "sel": ""}, {'name': 'Kalpetta', "sel": ""}, {'name': 'Mananthavadi', "sel": ""}, 
              {'name': 'Sultan Bathery', "sel": ""}, {'name': 'Thalassery', "sel": ""}, {'name': 'Taliparamba', "sel": ""}, 
              {'name': 'Payyanur', "sel": ""}, {'name': 'Mattannur', "sel": ""}, {'name': 'Koothuparamba', "sel": ""}, 
              {'name': 'Anthoor', "sel": ""}, {'name': 'Iritty', "sel": ""}, {'name': 'Panoor', "sel": ""}, 
              {'name': 'Sreekandapuram', "sel": ""}, {'name': 'Kasaragod', "sel": ""}, {'name': 'Kanhangad', "sel": ""}, 
              {'name': 'Nileshwaram', "sel": ""}, {'name': 'Vaikom', "sel": ""}, {'name': 'Ettumanoor', "sel": ""}, 
              {'name': 'Erattupetta', "sel": ""}, {'name': 'Thodupuzha', "sel": ""}, {'name': 'Kattappana', "sel": ""}, 
              {'name': 'Thripunithura', "sel": ""}, {'name': 'Thrikkakara', "sel": ""}, {'name': 'Kalamassery', "sel": ""}, 
              {'name': 'Perumbavoor', "sel": ""}, {'name': 'Aluva', "sel": ""}, {'name': 'Muvattupuzha', "sel": ""}, 
              {'name': 'Kothamangalam', "sel": ""}, {'name': 'North Paravoor', "sel": ""}, {'name': 'Angamaly', "sel": ""}, 
              {'name': 'Maradu', "sel": ""}, {'name': 'Eloor', "sel": ""}, {'name': 'Piravom', "sel": ""}, 
              {'name': 'Koothattukulam', "sel": ""}, {'name': 'Irinjalakuda', "sel": ""}, {'name': 'Kunnamkulam', "sel": ""}, 
              {'name': 'Chalakudy', "sel": ""}]  # Replace with the full list of cities

    temp = maxt = wspd = cloudcover = percip = humidity = pred = None  # Default values

    if request.method == "POST":
        try:
            cityname = request.form["city"]
            print(f"Selected city: {cityname}")
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
            print(f"Latitude: {lat}, Longitude: {lon}")

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