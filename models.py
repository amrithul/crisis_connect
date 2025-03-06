from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class DonationRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    help_type = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(50), default="Pending")

class Donation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    donor_name = db.Column(db.String(100), nullable=False)
    donation_type = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(50), default="Pending")