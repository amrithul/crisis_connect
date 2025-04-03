from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
from models import db, SelectedLocation  # Import SQLAlchemy models
import requests
from datetime import datetime
  # Import your prediction functions

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'  # Change to your actual DB URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

scheduler = BackgroundScheduler()

def check_flood_risk():
    from app import get_weather_data, predict_flood
    with app.app_context():  # Ensure database access in Flask context
        locations = SelectedLocation.query.all()
        for location in locations:
            weather_data = get_weather_data(location.latitude, location.longitude)
            if weather_data is not None:
                flood_risk = predict_flood(weather_data)
                location.flood_risk = "Unsafe" if flood_risk == 0 else "Safe"
                location.last_checked = datetime.utcnow()
        
        db.session.commit()
        print("✅ Flood risk updated for all locations.")

# Schedule the task to run every 4 hours
scheduler.add_job(check_flood_risk, "interval", hours=4)
scheduler.start()
