import logging
import json

import numpy as np
from PIL import Image
from celery.task.control import inspect
from celery import Task
from flask import Flask, request, abort, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

from celery_queue import app
from tasks import predict_vgg16, predict_mobilenet, predict_review_sentiment, predict_deeplab, predict_inception

i = inspect(app=app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__package__)

#TODO: use app.config
CELERY_BROKER_URL = 'redis://localhost'
CELERY_RESULT_BACKEND = 'redis://localhost'
ERROR_NO_IMAGE = 'Please provide an image'
ERROR_NO_TEXT = 'Please provide some text'

flask_app = Flask(__name__)
socketio = SocketIO(flask_app)
CORS(flask_app)

def handle_image(request):
    if 'file' not in request.files:
        abort(400, ERROR_NO_IMAGE)

    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        abort(400, ERROR_NO_IMAGE)

    img = Image.open(file)
    return np.array(img), request.form.get('sessionId')


def handle_text(request):
    if 'text' not in request.form:
        abort(400, ERROR_NO_TEXT)

    text = request.form['text']
    if len(text) < 1:
        abort(400, ERROR_NO_TEXT)

    return text, request.form.get('sessionId')


@flask_app.route('/notify_client', methods=['post'])
def notify_client_route():
    print(request.form)
    session_id = request.form.get('sessionId')
    data = request.form.get('data')
    socketio.emit('finished_job', {'data': data}, room=session_id)

    # TODO: proper response
    return "ok"


@flask_app.route('/status', methods=['get'])
def status_route():
    print(i.reserved())
    print(i.active())
    return "test"


@flask_app.route('/vgg16', methods=['POST'])
def vgg_route():
    img, session_id = handle_image(request)
    job = predict_vgg16.delay(img, session_id)
    return json.dumps({'jobId': job.id})


@flask_app.route('/mobilenet', methods=['POST'])
def mobilenet_route():
    img, session_id = handle_image(request)
    job = predict_mobilenet.delay(img, session_id)
    return json.dumps({'jobId': job.id})


@flask_app.route('/inception', methods=['POST'])
def inception_route():
    img, session_id = handle_image(request)
    job = predict_inception.delay(img, session_id)
    return json.dumps({'jobId': job.id})


@flask_app.route('/review-sentiment', methods=['POST'])
def review_sentiment_route():
    text, session_id = handle_text(request)
    job = predict_review_sentiment.delay(text, session_id)

    return json.dumps({'jobId': job.id})


@flask_app.route('/deeplab', methods=['POST'])
def deeplab_route():
    img, session_id = handle_image(request)
    job = predict_deeplab.delay(img, session_id)
    return json.dumps({'jobId': job.id})


@flask_app.route('/outputs/<path:path>')
def serve_images(path):
    return send_from_directory('outputs', path)

socketio.run(flask_app, debug=False, host='0.0.0.0', port=8091)
