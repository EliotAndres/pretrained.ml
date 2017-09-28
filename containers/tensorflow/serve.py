from flask import Flask, request, redirect, url_for, abort
from flask_cors import CORS
from PIL import Image

import logging

from models import VGG16Wrapper, MobileNetWrapper, InceptionV3Wrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__package__)

ERROR_NO_IMAGE = 'Please provide an image'

app = Flask(__name__)
CORS(app)

vgg16 = VGG16Wrapper()
mobilenet = MobileNetWrapper()
inception = InceptionV3Wrapper()


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

app.run(debug=False, host='0.0.0.0', port=8091)
