import json

import requests
from celery import Task
from celery.signals import worker_process_init

from celery_queue import app, redis_instance
from models import DeeplabWrapper, ReviewSentimentWrapper, MobileNetWrapper, VGG16Wrapper, InceptionV3Wrapper
i = app.control.inspect()

from celery.signals import after_task_publish, task_postrun


BASE_URL = 'http://web:8091/'

@after_task_publish.connect
def task_sent_handler(sender=None, headers=None, body=None, **kwargs):
    # information about task are located in headers for task messages
    # using the task protocol version 2.
    info = headers if 'task' in headers else body

    task_id = info['id']
    redis_instance.rpush('task_queue', task_id)

@task_postrun.connect
def task_run_handler(sender=None, task_id=None, task=None, args=None, retval=None,
                      kwargs=None, **kwds):
    redis_instance.lrem('task_queue', 1, task_id)

    data, session_id = retval
    # TODO: url in params
    requests.post(BASE_URL + 'notify_client',
                  data={'session_id': session_id, 'predictions': data, 'task_id': task_id})


vgg16_model = None
mobilenet_model = None
review_sentiment_model = None
deeplab_model = None
inception_model = None


@worker_process_init.connect()
def init_models(**_):
    global vgg16_model
    vgg16_model = VGG16Wrapper()

    global mobilenet_model
    mobilenet_model = MobileNetWrapper()

    global inception_model
    inception_model = InceptionV3Wrapper()

    global review_sentiment_model
    review_sentiment_model = ReviewSentimentWrapper()

    global deeplab_model
    deeplab_model = DeeplabWrapper()


@app.task
def predict_vgg16(img, session_id):
    predictions = vgg16_model.predict(img)
    return predictions, session_id


@app.task
def predict_mobilenet(img, session_id):
    predictions = mobilenet_model.predict(img)
    return predictions, session_id


@app.task
def predict_inception(img, session_id):
    predictions = inception_model.predict(img)
    return predictions, session_id


@app.task
def predict_review_sentiment(text, session_id):
    predictions = review_sentiment_model.predict(text)
    return predictions, session_id


@app.task
def predict_deeplab(img, session_id):
    predictions = deeplab_model.predict(img)
    return predictions, session_id
