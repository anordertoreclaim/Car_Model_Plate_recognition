import os
import numpy as np
from aocr.model.model import Model
import tensorflow as tf
import cv2


MAX_WIDTH, MAX_HEIGHT = 250, 125
CHECKPOINTS_DIR = 'models/Attention-OCR_car_plate_recognition/'
GPU_ID = -1

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
model = Model(
        phase='predict',
        visualize=False,
        output_dir='',
        batch_size=32,
        initial_learning_rate=1.0,
        steps_per_checkpoint=1500,
        model_dir=CHECKPOINTS_DIR,
        target_embedding_size=10,
        attn_num_hidden=128,
        attn_num_layers=2,
        clip_gradients=True,
        max_gradient_norm=5.0,
        session=sess,
        load_model=True,
        gpu_id=GPU_ID,
        use_gru=False,
        use_distance=True,
        max_image_width=MAX_WIDTH,
        max_image_height=MAX_HEIGHT,
        max_prediction_length=9,
        channels=1)

# def recognize(filename):
#     img = cv2.imread(filename)
#     if len(img.shape) == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     h, w = img.shape[0:2]

#     resize_ratio_h = 1
#     resize_ratio_w = 1

#     if h > MAX_HEIGHT:
#         resize_ratio_h = MAX_HEIGHT / h

#     if w > MAX_WIDTH:
#         resize_ratio_w = MAX_WIDTH / w

#     resize_ratio = min(resize_ratio_h, resize_ratio_w)
#     if resize_ratio != 1:
#         new_h, new_w = int(np.round(h * resize_ratio)), int(np.round(w * resize_ratio))
#         img = cv2.resize(img, (new_w, new_h))

#     h, w = img.shape[0:2]
#     top, bottom = np.floor((MAX_HEIGHT - h) / 2).astype(int), np.ceil((MAX_HEIGHT - h) / 2).astype(int)
#     left, right = np.floor((MAX_WIDTH - w) / 2).astype(int), np.ceil((MAX_WIDTH - w) / 2).astype(int)
#     img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])

#     img_bytes = cv2.imencode('.png', img_padded)[1].tostring()
#     text, prob = model.predict(img_bytes)
#     dict = {'У':'Y', 'Х':'X', 'Н':'H', 'К':'K', 'Е':'E', 'М':'M', 'С':'C', 'О':'O', 'В':'B', 'А':'A', 'Т':'T', 'Р':'P'}
#     for k,v in dict.items():
#         text = text.replace(v,k)

#     return text, prob


def recognize(filename):
    img = cv2.imread(filename)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[0:2]

    resize_ratio_h = MAX_HEIGHT / h
    resize_ratio_w = MAX_WIDTH / w

    resize_ratio = min(resize_ratio_h, resize_ratio_w)
    new_h, new_w = int(np.round(h * resize_ratio)), int(np.round(w * resize_ratio))
  
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_CUBIC)

    h, w = img.shape[0:2]
    top, bottom = np.floor((MAX_HEIGHT - h) / 2).astype(int), np.ceil((MAX_HEIGHT - h) / 2).astype(int)
    left, right = np.floor((MAX_WIDTH - w) / 2).astype(int), np.ceil((MAX_WIDTH - w) / 2).astype(int)
    img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])

    img_bytes = cv2.imencode('.png', img_padded)[1].tostring()
    text, prob = model.predict(img_bytes)
    dict = {'У':'Y', 'Х':'X', 'Н':'H', 'К':'K', 'Е':'E', 'М':'M', 'С':'C', 'О':'O', 'В':'B', 'А':'A', 'Т':'T', 'Р':'P'}
    for k,v in dict.items():
        text = text.replace(v,k)

    return text, prob