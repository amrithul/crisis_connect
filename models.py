from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class DonationRequest(db.Model):
    __tablename__ = 'donation_request'
    __table_args__ = {'extend_existing': True}  # Add this line

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    help_type = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(50), default="Pending")
    donated_item = db.Column(db.String(100))  # New column for donated item
    timestamp = db.Column(db.DateTime, default=datetime.now)

class Donation(db.Model):
    __tablename__ = 'donation'
    __table_args__ = {'extend_existing': True}  # Add this line

    id = db.Column(db.Integer, primary_key=True)
    donor_name = db.Column(db.String(100), nullable=False)
    donation_type = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(50), default="Pending")