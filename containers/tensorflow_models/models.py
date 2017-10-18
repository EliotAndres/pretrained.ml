# In order to import models without touching their code
# We add them to the path in order to import them as modules
import os, sys

reviews_path = os.path.abspath('./reviews')
sys.path.insert(0, reviews_path)
deeplab_path = os.path.abspath('./deeplab_resnet')
sys.path.insert(0, deeplab_path)

# For VGG16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg, \
    decode_predictions as decode_predictions_vgg

# For MobileNet
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as preprocess_input_mobilenet, \
    decode_predictions as decode_predictions_mobilenet

# For InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception, \
    decode_predictions as decode_predictions_inception

# For sentiment analysis
from encoder import Model as SentimentModel

# For deeplab
from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label
from PIL import Image
import tensorflow as tf

from keras.preprocessing import image as keras_image
import numpy as np

import json
import logging
import uuid
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__package__)


class VGG16Wrapper(object):
    def __init__(self):
        logger.info('Loading vgg16')
        self.model = VGG16(weights='imagenet')

    def predict(self, img):
        """ # Arguments
                img: a numpy array

            # Returns
                A dict containing predictions
            """
        img = Image.fromarray(img)
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

    def predict(self, img):
        """ # Arguments
                img: a numpy array

            # Returns
                A dict containing predictions
            """
        img = Image.fromarray(img)
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

    def predict(self, img):
        """ # Arguments
                img: a numpy array

            # Returns
                A dict containing predictions
            """
        img = Image.fromarray(img)
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
        self.g = tf.Graph()
        with self.g.as_default():
            current_directory = os.getcwd()

            # Necessary as the model is imported with relative path
            os.chdir(reviews_path)
            self.model = SentimentModel()
            os.chdir(current_directory)

    def predict(self, text):
        """ # Arguments
                text: a string to process

        # Returns
            A dict containing predictions
        """
        text_features = self.model.transform([text])
        # For more info https://github.com/openai/generating-reviews-discovering-sentiment/issues/2
        sentiment = text_features[0, 2388]

        return json.dumps({'sentiment': str(sentiment)})


class DeeplabWrapper(object):
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    NUM_CLASSES = 21

    def __init__(self):
        logger.info('Loading Deeplab')
        self.g = tf.Graph()
        with self.g.as_default():
            self.image_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 3))
            self.net = DeepLabResNetModel({'data': self.image_placeholder}, is_training=False,
                                          num_classes=self.NUM_CLASSES)

            restore_var = tf.global_variables()

            # Set up TF session and initialize variables.
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            init = tf.global_variables_initializer()

            self.sess.run(init)

            # Load weights.
            loader = tf.train.Saver(var_list=restore_var)
            loader.restore(self.sess, './deeplab_resnet/deeplab_resnet.ckpt')

    def predict(self, img):
        """ # Arguments
                img: a numpy array

            # Returns
                The url to an image with the segmentation
            """

        with self.g.as_default():
            img = Image.fromarray(img)
            # RGB -> BGR
            b, g, r = img.split()
            img = Image.merge("RGB", (r, g, b))
            img -= self.IMG_MEAN

            # Predictions.
            raw_output = self.net.layers['fc1_voc12']
            raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2, ])
            raw_output_up = tf.argmax(raw_output_up, axis=3)
            self.pred = tf.expand_dims(raw_output_up, dim=3)

            preds = self.sess.run(self.pred, feed_dict={self.image_placeholder: np.expand_dims(img, axis=0)})

        msk = decode_labels(preds, num_classes=self.NUM_CLASSES)
        im = Image.fromarray(msk[0])

        filename = str(uuid.uuid4()) + '.png'
        save_dir = './outputs'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filename)
        im.save(save_path)

        return json.dumps({'output': filename})
