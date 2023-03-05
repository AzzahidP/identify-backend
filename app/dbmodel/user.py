from app import db
from datetime import datetime

class User(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    full_name = db.Column(db.String(250), nullable=False)
    identity_number = db.Column(db.BigInteger, nullable=False, unique=True)
    vector = db.Column(db.String(7500), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    birth = db.Column(db.String(80), nullable=False)
    address = db.Column(db.String(250), nullable=False)
    email = db.Column(db.String(250), index=True, unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __represent__(self):
        return '<User {}>'.format(self.name)