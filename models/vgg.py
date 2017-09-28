# For VGG16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg,\
    decode_predictions as decode_predictions_vgg

# For MobileNet
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import preprocess_input as preprocess_input_mobilenet,\
    decode_predictions as decode_predictions_mobilenet

from keras.preprocessing import image
import numpy as np
from flask import Flask, request, redirect, url_for, abort
import cv2
from PIL import Image

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ERROR_NO_IMAGE = 'Please provide an image'


app = Flask(__name__)

logger.info('Loading vgg16')
vgg16_model = VGG16(weights='imagenet')
logger.info('Loading mobilenet')
mobilenet_model = MobileNet(weights='imagenet')

#def handle_image(model, request):

@app.route('/vgg16', methods=['POST'])
def vgg():
    if 'file' not in request.files:
        abort(400, ERROR_NO_IMAGE)
        return

    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        abort(400, ERROR_NO_IMAGE)

    img = Image.open(file)
    img = img.resize((224, 224))
    print(img.size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_vgg(x)

    features = vgg16_model.predict(x)
    predictions = decode_predictions_vgg(features)[0]
    clean_predictions = [{'score': str(k), 'class': j} for (i, j, k) in predictions]

    return json.dumps(clean_predictions)

@app.route('/mobilenet', methods=['POST'])
def mobilenet():
    if 'file' not in request.files:
        abort(400, ERROR_NO_IMAGE)
        return

    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        abort(400, ERROR_NO_IMAGE)

    img = Image.open(file)
    img = img.resize((224, 224))
    print(img.size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_mobilenet(x)

    features = mobilenet_model.predict(x)
    predictions = decode_predictions_mobilenet(features)[0]
    clean_predictions = [{'score': str(k), 'class': j} for (i, j, k) in predictions]

    return json.dumps(clean_predictions)

app.run(debug=False, host='0.0.0.0', port=8091)