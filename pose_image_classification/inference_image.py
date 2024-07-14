from fastai.learner import load_learner
from fastai.vision import *
import os
import json


def list_folders(directory):
    folders =  [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    folders.sort()
    return folders


def predict_image_class(image_path, classes_json):

    learn = load_learner(os.path.join('static', 'image_classification_model.pkl'))
    cls, ss, d = learn.predict(image_path)
    with open(classes_json, 'r') as f:
        classes_names = json.load(f)
    # classes_names = list_folders('static/data/training')
    confs = [float(conf) for conf in d]
    return cls, confs, classes_names
