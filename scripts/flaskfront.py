import json
from flask import Flask, escape, request, jsonify
# from label_image import *
import label_image

#you have to pip3 install flask and Flask-JSON

app = Flask(__name__)
# json = FlaskJSON(app)

@app.route('/')
def hello():
    return "<h1>Beautiful Flaske Server!</h1>" + "\n<h2>enter</h2>"

def label(name):
    # label_image
    return

@app.route('/define')
def define():
    return "<h1>defining an image</h1>"

if __name__ == '__main__':
    app.run(debug=True)
