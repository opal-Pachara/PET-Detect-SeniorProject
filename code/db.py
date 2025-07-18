from flask_pymongo import PyMongo

mongo = PyMongo()

def init_db(app):
    app.config["MONGO_URI"] = "mongodb+srv://s6404062636412:0606@pet.tacvdh9.mongodb.net/pet?retryWrites=true&w=majority&appName=pet"
    mongo.init_app(app)