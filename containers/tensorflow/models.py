# For VGG16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg,\
    decode_predictions as decode_predictions_vgg

# For MobileNet
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as preprocess_input_mobilenet,\
    decode_predictions as decode_predictions_mobilenet

# For InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception,\
    decode_predictions as decode_predictions_inception

# For reviews sentiment analysis
import os, sys
path = os.path.abspath('./reviews')
os.chdir(path)
sys.path.insert(0, path)

from encoder import Model as SentimentModel

from keras.preprocessing import image as keras_image
import numpy as np

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__package__)


class VGG16Wrapper(object):
    def __init__(self):
        logger.info('Loading vgg16')
        self.model = VGG16(weights='imagenet')

    """ # Arguments
            img: a PIL image instance

        # Returns
            A dict containing predictions
        """
    def predict(self, img):
        img = img.resize((224, 224))
        x = keras_image.img_to_array(img)[:, :, :3]
        x = np.expand_dims(x, axis=0)
        x = preprocess_input_vgg(x)

        features = self.model.predict(x)
        predictions = decode_predictions_vgg(features)[0]
        clean_predictions = [{'score': str(k), 'class': j} for (i, j, k) in predictions]

        return json.dumps(clean_predictions)


class MobileNetWrapper(object):
    def __init__(self):
        logger.info('Loading MobileNet')
        self.model = MobileNet(weights='imagenet')

    """ # Arguments
            img: a PIL image instance

        # Returns
            A dict containing predictions
        """
    def predict(self, img):
        img = img.resize((224, 224))
        x = keras_image.img_to_array(img)[:, :, :3]
        x = np.expand_dims(x, axis=0)
        x = preprocess_input_mobilenet(x)

        features = self.model.predict(x)
        predictions = decode_predictions_mobilenet(features)[0]
        clean_predictions = [{'score': str(k), 'class': j} for (i, j, k) in predictions]

        return json.dumps(clean_predictions)


class InceptionV3Wrapper(object):
    def __init__(self):
        logger.info('Loading Inception V3')
        self.model = InceptionV3(weights='imagenet')

    """ # Arguments
            img: a PIL image instance

        # Returns
            A dict containing predictions
        """
    def predict(self, img):
        img = img.resize((224, 224))
        x = keras_image.img_to_array(img)[:, :, :3]
        x = np.expand_dims(x, axis=0)
        x = preprocess_input_inception(x)

        features = self.model.predict(x)
        predictions = decode_predictions_inception(features)[0]
        clean_predictions = [{'score': str(k), 'class': j} for (i, j, k) in predictions]

        return json.dumps(clean_predictions)

class ReviewSentimentWrapper(object):
    def __init__(self):
        logger.info('Loading Review sentiment')
        self.model = SentimentModel()

    def predict(self, text):
        text = [' This is very bad', ' This is very Good']
        text_features = self.model.transform([text])
        # For more info https://github.com/openai/generating-reviews-discovering-sentiment/issues/2
        sentiment = text_features[0, 2388]

        return json.dumps({'sentiment': str(sentiment)})

