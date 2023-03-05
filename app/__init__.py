from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from dbconfig import AppConfig

app = Flask(__name__)
cors = CORS(app)
app.config.from_object(AppConfig)


db = SQLAlchemy(app)
migrate = Migrate(app, db)

from app.dbmodel import user
from app import routes