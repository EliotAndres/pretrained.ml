import logging
import json

import numpy as np
from PIL import Image
from flask import Flask, request, abort, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

from celery_queue import app, redis_instance
import config
from tasks import predict_vgg16, predict_mobilenet, predict_review_sentiment, \
    predict_deeplab, predict_inception, predict_ssd_inception

i = app.control.inspect()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__package__)
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)

flask_app = Flask(__name__)

socketio = SocketIO(flask_app)
CORS(flask_app)

def handle_image(request):
    if 'file' not in request.files:
        abort(400, config.ERROR_NO_IMAGE)

    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        abort(400, config.ERROR_NO_IMAGE)

    img = Image.open(file)
    img.thumbnail(config.MAX_SIZE, Image.ANTIALIAS)
    return np.array(img)[:, :, :3], request.form.get('sessionId')


def handle_text(request):
    if 'text' not in request.form:
        abort(400, config.ERROR_NO_TEXT)

    text = request.form['text']
    if len(text) < 1:
        abort(400, config.ERROR_NO_TEXT)

    return text, request.form.get('sessionId')


def broadcast_queue_status():
    # Broadcast queue status to all clients
    task_ids = redis_instance.lrange('task_queue', 0, -1)
    task_ids = [task.decode() for task in task_ids]
    tasks_json = json.dumps(task_ids)
    socketio.emit('queue_status', {'data': tasks_json}, broadcast=True)
    return tasks_json


@flask_app.route('/notify_client', methods=['post'])
def notify_client_route():
    session_id = request.form.get('session_id')
    predictions = request.form.get('predictions')
    task_id = request.form.get('task_id')

    if session_id is not None:
        socketio.emit('finished_job', {'predictions': predictions, 'taskId': task_id}, room=session_id)

    broadcast_queue_status()
    return '200 OK'


@flask_app.route('/status', methods=['get'])
def status_route():
    tasks_json = broadcast_queue_status()
    return tasks_json


@flask_app.route('/vgg16', methods=['POST'])
def vgg_route():
    img, session_id = handle_image(request)
    job = predict_vgg16.delay(img, session_id)
    return json.dumps({'taskId': job.id})


@flask_app.route('/mobilenet', methods=['POST'])
def mobilenet_route():
    img, session_id = handle_image(request)
    job = predict_mobilenet.delay(img, session_id)
    return json.dumps({'taskId': job.id})


@flask_app.route('/inception', methods=['POST'])
def inception_route():
    img, session_id = handle_image(request)
    job = predict_inception.delay(img, session_id)
    return json.dumps({'taskId': job.id})


@flask_app.route('/review-sentiment', methods=['POST'])
def review_sentiment_route():
    text, session_id = handle_text(request)
    job = predict_review_sentiment.delay(text, session_id)

    return json.dumps({'taskId': job.id})


@flask_app.route('/deeplab', methods=['POST'])
def deeplab_route():
    img, session_id = handle_image(request)
    job = predict_deeplab.delay(img, session_id)
    return json.dumps({'taskId': job.id})


@flask_app.route('/ssd-inception', methods=['POST'])
def faster_rcnn_route():
    img, session_id = handle_image(request)
    job = predict_ssd_inception.delay(img, session_id)
    return json.dumps({'taskId': job.id})


@flask_app.route('/outputs/<path:path>')
def serve_images(path):
    return send_from_directory('outputs', path)

logger.info('Web server starting')
socketio.run(flask_app, debug=False, host='0.0.0.0', port=8091)

