import matplotlib.pyplot as plt
from fastai.imports import *
from fastai.torch_core import *
from fastai.learner import *
import inspect
from fastai.data.all import *
from fastai.vision.all import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_image_classification(image_dataset):

    dls = ImageDataLoaders.from_folder(image_dataset, train='training', valid='validation', item_tfms=RandomResizedCrop(128, min_scale=0.35),
                                       batch_tfms=Normalize.from_stats(*imagenet_stats), size=224, bs=16)

    # Training: resne152
    # For the training, using resnet152 and batch size of 32.
    epochs = 10
    learn = vision_learner(dls, models.resnet152,metrics=[accuracy,error_rate,Precision(average='micro')], cbs=[ShowGraphCallback()])

    # learn = vision_learner(dls, models.resnet152,metrics=[accuracy,Precision(average='micro'),Recall(average='micro'),FBeta(beta=1,average='micro'),RocAuc(),error_rate],cbs=[ShowGraphCallback()])
    # learn = vision_learner(dls, models.densenet201,metrics=[accuracy,Precision(average='micro'),Recall(average='micro'),FBeta(beta=1,average='micro'),RocAuc(),error_rate],cbs=[ShowGraphCallback()])

    learn.fit_one_cycle(epochs)
    preds,y, loss = learn.get_preds(with_loss=True)
    acc = accuracy(preds, y)
    acc = acc * 100
    learn.export('../../static/image_classification_model.pkl')

    return str(float(acc))
