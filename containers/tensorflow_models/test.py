from models import DetectionApiWrapper
import cv2
from PIL import Image


detect = DetectionApiWrapper()

im = cv2.imread('elephant.png')


resp = detect.predict(im)

print(resp)

#from .reviews.encoder import Model as SentimentModel
