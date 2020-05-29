from efficientnet_pytorch import EfficientNet
import torch
from torch import nn

from PIL import Image
import numpy as np
from albumentations import (Rotate, RandomBrightness, RandomContrast,
                            Cutout, Compose, OneOf, Resize, Normalize)
from albumentations.pytorch.transforms import ToTensor

from torch.nn.parameter import Parameter
from typing import List
import math
from torch.nn import init
import torch.nn.functional as F


import cv2
import matplotlib.pyplot as plt

    
class LazyLoadModule(nn.Module):
    """Lazy buffer/parameter loading using load_state_dict_pre_hook

    Define all buffer/parameter in `_lazy_buffer_keys`/`_lazy_parameter_keys` and
    save buffer with `register_buffer`/`register_parameter`
    method, which can be outside of __init__ method.
    Then this module can load any shape of Tensor during de-serializing.

    Note that default value of lazy buffer is torch.Tensor([]), while lazy parameter is None.
    """
    _lazy_buffer_keys: List[str] = []  # It needs to be override to register lazy buffer
    _lazy_parameter_keys: List[str] = []  # It needs to be override to register lazy parameter

    def __init__(self):
        super(LazyLoadModule, self).__init__()
        for k in self._lazy_buffer_keys:
            self.register_buffer(k, torch.tensor([]))
        for k in self._lazy_parameter_keys:
            self.register_parameter(k, None)
        self._register_load_state_dict_pre_hook(self._hook)

    def _hook(self, state_dict, prefix, local_metadata, strict, missing_keys,
              unexpected_keys, error_msgs):
        for key in self._lazy_buffer_keys:
            self.register_buffer(key, state_dict[prefix + key])

        for key in self._lazy_parameter_keys:
            self.register_parameter(key, Parameter(state_dict[prefix + key]))


class LazyLinear(LazyLoadModule):
    """Linear module with lazy input inference

    `in_features` can be `None`, and it is determined at the first time of forward step dynamically.
    """

    __constants__ = ['bias', 'in_features', 'out_features']
    _lazy_parameter_keys = ['weight']

    def __init__(self, in_features, out_features, bias=True):
        super(LazyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if in_features is not None:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
            self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.weight is None:
            self.in_features = input.shape[-1]
            self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
            self.reset_parameters()

            # Need to send lazy defined parameter to device...
            self.to(input.device)
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class PlateDetector(nn.Module):
    def __init__(self):
        super(PlateDetector, self).__init__()
        self.backbone = EfficientNet.from_name('efficientnet-b4')
        # self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.15)
        self.dropout3 = nn.Dropout(0.15)
        self.linear = LazyLinear(in_features=None, out_features=1024)

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 4)

        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x = self.backbone.extract_features(img).view(img.shape[0], -1)
        x = self.dropout1(x)

        x = self.linear(x)
        x = self.gelu(x)
        x = self.dropout2(x)

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class PlateDetectorModel(object):
    def __init__(self, model_path):
        self.IMAGE_SIZE = 256
        self.aug_transform = Compose([Resize(self.IMAGE_SIZE, self.IMAGE_SIZE), Normalize(), ToTensor()])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PlateDetector()
        self.model = self.model.to(self.device)

        cp = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(cp['state_dict'])
        self.model.eval()

    def __call__(self, img_path):
        image = np.asarray(Image.open(img_path).convert('RGB'))
        img = image.copy()
            
        
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
  
        has_rect, x, y, w, h = find_rectangle(image)

        if not has_rect:
            image = np.asarray(Image.open(img_path).convert('RGB'))
            image = self.aug_transform(image=image)['image']
            image = image.to(self.device)
            image = image.reshape(-1, 3, self.IMAGE_SIZE, self.IMAGE_SIZE)

            # print(image.shape)
            with torch.no_grad():
                out = self.model(image)
                [x, y, w, h] = out[0].cpu().numpy()

        # plt.imshow(img)
        # plt.show()

        h_i, w_i, _ = img.shape
        # print(w_i, h_i)

        img_c = img.copy()
        cv2.rectangle(img_c, (int(x * w_i - w * w_i / 2), int(y * h_i - h * h_i / 2)),
                      (int(x * w_i + w * w_i / 2), int(y * h_i + h * h_i / 2)), (255, 255, 0), 5)
        # plt.imshow(img_c)
        # plt.show()

        img_res = img[max(0, int(y * h_i - h * h_i / 2)):min(h_i - 1, int(y * h_i + h * h_i / 2)),
                  max(0, int(x * w_i - w * w_i / 2)):min(w_i, int(x * w_i + w * w_i / 2)), :]
        # plt.imshow(img_res)
        # plt.show()
        return img_res

my_model = PlateDetectorModel('models/EfficientNet_car_plate_detection/model_e30.pth')

def find_rectangle(image):
    image_c = image.copy()
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    sat = img[:,:,1]
    
    _,sat = cv2.threshold(sat,65,255,cv2.THRESH_BINARY)
    
    # plt.imshow(sat, cmap='gray')
    # plt.show()

    kernel = np.ones((3,3),np.uint8)
    
    sat = cv2.morphologyEx(sat, cv2.MORPH_ERODE, kernel)
    sat = cv2.morphologyEx(sat, cv2.MORPH_DILATE, kernel)
    # plt.imshow(sat, cmap='gray')
    # plt.show()

    
    h, w = sat.shape 
    left_x, top_y = 9999, 9999
    right_x, bot_y = -1, -1

    for i in range(h):
      for j in range(w):
        if sat[i,j]!=0:
          
          if j < left_x:
            left_x = j
          
          if j > right_x:
            right_x = j
          
          if i < top_y:
            top_y = i
          if i > bot_y:
            bot_y = i 

    if left_x == 9999:
      return False, None, None, None, None
    # print((left_x, top_y), (right_x, bot_y))

    # left_x, right_x = left_x + int((right_x-left_x)*0.1), right_x - int((right_x-left_x)*0.1)
    # top_y, bot_y = top_y + int((bot_y-top_y)*0.25) , bot_y - int((bot_y-top_y)*0.25)
    

    k = int((bot_y-top_y)*0.25)
    
    left_x, right_x = left_x + k, right_x - k
    top_y, bot_y = top_y + k , bot_y - k

    

    x = (left_x + (right_x-left_x)/2)/w
    y = (top_y + (bot_y-top_y)/2)/h
    w = (right_x-left_x)/w
    h = (bot_y-top_y)/h
    return True, x, y, w, h
    

def detect(image_path):
    img = my_model(image_path)
    cv2.imwrite('X000XX000.jpg',img)
    