from flask import Flask, request, redirect, url_for, abort, send_from_directory
from flask_cors import CORS
from PIL import Image

import logging

from models import VGG16Wrapper, MobileNetWrapper,\
    InceptionV3Wrapper, ReviewSentimentWrapper, DeeplabWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__package__)

ERROR_NO_IMAGE = 'Please provide an image'
ERROR_NO_TEXT = 'Please provide some text'

app = Flask(__name__)
CORS(app)

vgg16 = VGG16Wrapper()
mobilenet = MobileNetWrapper()
inception = InceptionV3Wrapper()
review_sentiment = ReviewSentimentWrapper()
deeplab = DeeplabWrapper()


def handle_image(request):
    if 'file' not in request.files:
        abort(400, ERROR_NO_IMAGE)

    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        abort(400, ERROR_NO_IMAGE)

    img = Image.open(file)
    return img


def handle_text(request):
    if 'text' not in request.form:
        abort(400, ERROR_NO_TEXT)

    text = request.form['text']
    if len(text) < 1:
        abort(400, ERROR_NO_TEXT)

    return text


@app.route('/vgg16', methods=['POST'])
def vgg_route():
    img = handle_image(request)
    predictions = vgg16.predict(img)

    return predictions

@app.route('/mobilenet', methods=['POST'])
def mobilenet_route():
    img = handle_image(request)
    predictions = mobilenet.predict(img)

    return predictions

@app.route('/inception', methods=['POST'])
def inception_route():
    img = handle_image(request)
    predictions = inception.predict(img)

    return predictions

@app.route('/review-sentiment', methods=['POST'])
def review_sentiment_route():
    text = handle_text(request)
    predictions = review_sentiment.predict(text)

    return predictions

@app.route('/deeplab', methods=['POST'])
def deeplab_route():
    img = handle_image(request)
    predictions = deeplab.predict(img)

    return predictions

@app.route('/outputs/<path:path>')
def send_js(path):
    return send_from_directory('outputs', path)

app.run(debug=False, host='0.0.0.0', port=8091)
