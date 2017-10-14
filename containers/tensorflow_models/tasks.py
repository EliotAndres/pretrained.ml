import requests
from celery import Task
from celery.signals import worker_process_init

from celery_queue import app
from models import DeeplabWrapper, ReviewSentimentWrapper, MobileNetWrapper, VGG16Wrapper, InceptionV3Wrapper


class CallbackTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        data, session_id = retval
        # TODO: url in params
        requests.post('http://localhost:8091/notify_client', data={'sessionId': session_id, 'data': data})


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


@app.task(base=CallbackTask)
def predict_vgg16(img, session_id):
    predictions = vgg16_model.predict(img)
    return predictions, session_id


@app.task(base=CallbackTask)
def predict_mobilenet(img, session_id):
    predictions = mobilenet_model.predict(img)
    return predictions, session_id


@app.task(base=CallbackTask)
def predict_inception(img, session_id):
    predictions = inception_model.predict(img)
    return predictions, session_id


@app.task(base=CallbackTask)
def predict_review_sentiment(text, session_id):
    predictions = review_sentiment_model.predict(text)
    return predictions, session_id


@app.task(base=CallbackTask)
def predict_deeplab(img, session_id):
    predictions = deeplab_model.predict(img)
    return predictions, session_id
