import os, sys
# In order to import models without touching their code
# We add them to the path in order to import them as modules

reviews_path = os.path.abspath('./reviews')
TF_MODELS_BASE_PATH = './tensorflow_models_repo'
paths = [reviews_path,
         os.path.abspath('./deeplab_resnet'),
         os.path.abspath(os.path.join(TF_MODELS_BASE_PATH, 'research')),
         os.path.abspath(os.path.join(TF_MODELS_BASE_PATH, 'research/slim')),
         ]
for path in paths:
    sys.path.insert(0, path)

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

# For Tensorflow model analysis
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

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
        self.graph = tf.Graph()
        with self.graph.as_default():
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
        self.graph = tf.Graph()
        with self.graph.as_default():
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

        with self.graph.as_default():
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

        filename = str(uuid.uuid4()) + '.jpg'
        save_dir = './outputs'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filename)
        im.save(save_path)

        return json.dumps({'output': filename})


class DetectionApiWrapper(object):
    PATH_TO_CKPT = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join(os.path.join(TF_MODELS_BASE_PATH,
                                               'research/object_detection/data'),
                                  'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    def __init__(self):
        logger.info('Loading Tensorflow Detection API')
        self.graph = tf.Graph()

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.NUM_CLASSES,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def predict(self, img):
        """ # Arguments
                img: a numpy array

            # Returns
                The url to an image with the bounding boxes
            """

        def load_image_into_numpy_array(image):
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)

        with self.graph.as_default():
            with tf.Session(graph=self.graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.graph.get_tensor_by_name('num_detections:0')
                image = Image.fromarray(img)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                im = Image.fromarray(image_np)
                filename = str(uuid.uuid4()) + '.jpg'
                save_dir = './outputs'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, filename)
                im.save(save_path)

                return json.dumps({'output': filename})
