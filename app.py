import flask
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import base64
import requests
import datetime
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from apscheduler.schedulers.background import BackgroundScheduler
from folium.plugins import HeatMap
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from models import db, DonationRequest, Donation
import threading
import time

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///donations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
alert_data = {"alerts": []} 
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

    
    heatmap_data = []

    for lat, lon in LOCATIONS:
        weather_data = get_weather_data(lat, lon)  # Fixed: Changed get_data to get_weather_data
        flood_risk = predict_flood(weather_data)  # Get flood prediction (0 or 1)

        # Assign intensity for heatmap (higher for flood-prone areas)
        intensity = 0.8 if flood_risk == 1 else 0.2
        heatmap_data.append({"lat": lat, "lon": lon, "intensity": intensity})

    return jsonify(heatmap_data)

data = [{'name':'Delhi', "sel": "selected"}, {'name':'Mumbai', "sel": ""}, {'name':'Kolkata', "sel": ""}, {'name':'Bangalore', "sel": ""}, {'name':'Chennai', "sel": ""}]
months = [{"name":"May", "sel": ""}, {"name":"June", "sel": ""}, {"name":"July", "sel": "selected"}]
cities = [{'name':'Delhi', "sel": "selected"}, {'name':'Mumbai', "sel": ""}, {'name':'Kolkata', "sel": ""}, {'name':'Bangalore', "sel": ""}, {'name':'Chennai', "sel": ""}, {'name':'New York', "sel": ""}, {'name':'Los Angeles', "sel": ""}, {'name':'London', "sel": ""}, {'name':'Paris', "sel": ""}, {'name':'Sydney', "sel": ""}, {'name':'Beijing', "sel": ""}]




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


@app.route('/index.html')
def index():
    global unsafe_locations
    if unsafe_locations:
        alert_message = "⚠️ Unsafe locations detected!"
    else:
        alert_message = "✅ All locations are currently safe."
    return render_template('index.html', alert_message=alert_message, unsafe_locations=unsafe_locations)

def check_all_locations():
    global unsafe_locations
    print(f"[{datetime.now()}] Running safety check for all locations...")


    city_coords = {
    "Tirurangadi": (11.0427, 75.9221),
    "Vatakara": (11.6080, 75.5917),
    "Koyilandy": (11.4426, 75.6950),
    "Mukkam": (11.3046, 75.9841),
    "Ramanattukara": (11.1796, 75.8414),
    "Feroke": (11.1793, 75.8414),
    "Payyoli": (11.5250, 75.6930),
    "Koduvally": (11.4410, 75.9320),
    "Kalpetta": (11.6086, 76.0827),
    "Mananthavadi": (11.8030, 76.0140),
    "Sultan Bathery": (11.6667, 76.2667),
    "Thalassery": (11.7500, 75.5000),
    "Taliparamba": (12.0500, 75.3500),
    "Payyanur": (12.0930, 75.2020),
    "Mattannur": (11.9300, 75.5700),
    "Koothuparamba": (11.8333, 75.5667),
    "Anthoor": (11.9870, 75.3760),
    "Iritty": (11.9833, 75.6833),
    "Panoor": (11.7500, 75.5667),
    "Sreekandapuram": (12.0500, 75.4500),
    "Kasaragod": (12.5000, 74.9900),
    "Kanhangad": (12.3100, 75.0800),
    "Nileshwaram": (12.2648, 75.1257),
    "Neyyattinkara": (8.3983, 77.0871),
    "Nedumangad": (8.6021, 77.0023),
    "Attingal": (8.6961, 76.8154),
    "Varkala": (8.7384, 76.7169),
    "Punalur": (9.0030, 76.9313),
    "Karunagappally": (9.0176, 76.5210),
    "Paravur": (8.7832, 76.7015),
    "Kottarakkara": (9.0062, 76.7768),
    "Thiruvalla": (9.3840, 76.5742),
    "Pathanamthitta": (9.2640, 76.7879),
    "Adoor": (9.1643, 76.7516),
    "Pandalam": (9.3206, 76.7392),
    "Alappuzha": (9.4981, 76.3388),
    "Kayamkulam": (9.1813, 76.5002),
    "Cherthala": (9.6841, 76.3391),
    "Mavelikara": (9.2583, 76.5568),
    "Chengannur": (9.3333, 76.6167),
    "Haripad": (9.2889, 76.4733),
    "Kottayam": (9.5916, 76.5222),
    "Changanacherry": (9.4420, 76.5440),
    "Pala": (9.7072, 76.6840),
    "Pattambi": (10.7832, 76.1831),
    "Shoranur": (10.7576, 76.2717),
    "Kodungallur": (10.2320, 76.1951),
    "Chittur": (10.6995, 76.7479),
    "Cherpulassery": (10.8783, 76.3128),
    "Chavakkad": (10.7000, 76.0500),
    "Guruvayoor": (10.5941, 76.0410),
    "Mannarkkad": (10.9910, 76.4600),
    "Ottapalam": (10.7700, 76.3776),
    "Wadakkanchery": (10.6500, 76.1167),
    "Palakkad": (10.7867, 76.6548),
    "Vaikom": (9.7481, 76.3964),
    "Ettumanoor": (9.6686, 76.5570),
    "Erattupetta": (9.6874, 76.7800),
    "Thodupuzha": (9.8920, 76.7184),
    "Kattappana": (9.7510, 77.1160),
    "Thripunithura": (9.9457, 76.3419),
    "Thrikkakara": (10.0280, 76.3264),
    "Kalamassery": (10.0480, 76.3083),
    "Perumbavoor": (10.1115, 76.4797),
    "Aluva": (10.1076, 76.3452),
    "Muvattupuzha": (9.9796, 76.5738),
    "Kothamangalam": (10.0615, 76.6297),
    "North Paravoor": (10.1411, 76.2296),
    "Angamaly": (10.1960, 76.3860),
    "Maradu": (9.9390, 76.3183),
    "Eloor": (10.0670, 76.2920),
    "Piravom": (9.8667, 76.5000),
    "Koothattukulam": (9.8333, 76.5833),
    "Irinjalakuda": (10.3428, 76.2110),
    "Kunnamkulam": (10.6500, 76.0667),
    "Chalakudy": (10.3000, 76.3317)
    }

    
    new_unsafe = []

    for city in city_coords.keys():
        try:
            lat, lon = city_coords[city]

            # Get weather
            weather_data = get_weather_data(lat, lon)
            if weather_data is None:
                print(f"No weather data for {city}")
                continue

            # Predict
            flood_risk = predict_flood(weather_data)
            print(f"Checked {city}: Flood risk = {flood_risk}")

            if flood_risk == 1:
                new_unsafe.append(city)
        except Exception as e:
            print(f"Error processing {city}: {e}")

    unsafe_locations = new_unsafe
    print(f"Unsafe locations updated: {unsafe_locations}")

# Schedule the function to run every 4 hours
scheduler = BackgroundScheduler()
scheduler.add_job(func=check_all_locations, trigger="interval", hours=4)
scheduler.start()

# Shutdown on exit
import atexit
atexit.register(lambda: scheduler.shutdown())


if __name__ == "__main__":
    check_all_locations()
    with app.app_context():
        db.create_all()
    app.run(debug=True)