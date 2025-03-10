import time
from datetime import datetime, timedelta
from models import db, DonationRequest  # Import from models.py
from app import app  # Import the Flask app

def delete_fulfilled_requests():
    with app.app_context():
        # Calculate the time threshold (1 hour ago)
        threshold_time = datetime.utcnow() - timedelta(hours=1)

        # Find and delete fulfilled requests older than 1 hour
        fulfilled_requests = DonationRequest.query.filter(
            DonationRequest.status == "Fulfilled",
            DonationRequest.fulfilled_time <= threshold_time
        ).all()

        for request in fulfilled_requests:
            db.session.delete(request)
            print(f"Deleted request: {request.id}")

        db.session.commit()

if __name__ == "__main__":
    delete_fulfilled_requests()