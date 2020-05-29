import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch import nn
from PIL import Image
from albumentations import (Compose, Resize, Normalize)
from albumentations.pytorch.transforms import ToTensor

import os
import numpy as np
from typing import List
import math


IMAGE_SIZE = 224
PATH_TO_WEIGHTS = 'models/EfficientNet_car_model_classification/model_e4.pth'


class LazyLoadModule(nn.Module):
    """Lazy buffer/parameter loading using load_state_dict_pre_hook

    Define all buffer/parameter in `_lazy_buffer_keys`/`_lazy_parameter_keys` and
    save buffer with `register_buffer`/`register_parameter`
    method, which can be outside of __init__ method.
    Then this module can load any shape of Tensor during de-serializing.

    Note that default value of lazy buffer is torch.Tensor([]), while lazy parameter is None.
    """
    _lazy_buffer_keys: List[str] = []     # It needs to be override to register lazy buffer
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

class CarsRecognizer(nn.Module):
  def __init__(self, n_classes):
    super(CarsRecognizer, self).__init__()
    self.backbone = EfficientNet.from_name('efficientnet-b5')
    self.avg_pooling = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(0.15)
    self.linear = LazyLinear(in_features=None, out_features=n_classes)

  def forward(self, img):
    features = self.backbone.extract_features(img)
    features = self.avg_pooling(features).squeeze()
    features = self.dropout(features)
    logits = self.linear(features)
    return logits

test_transform = Compose([Resize(IMAGE_SIZE, IMAGE_SIZE), Normalize(), ToTensor()])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CarsRecognizer(n_classes=11)
cp = torch.load(PATH_TO_WEIGHTS, map_location=device)
model.load_state_dict(cp['state_dict'])

def predict(image_path):
  id2cartype = {0: 'MAZDA_3_B',
                1: 'КIА_RIО_B',
                2: 'VОLКSWАGЕN_РОLО_B',
                3: 'VOLVO_ALLVOLVO_C',
                4: 'НУUNDАI_SОLАRIS_B',
                5: 'LADA_PRIORA_B',
                6: 'VОLКSWАGЕN_TIGUAN_B',
                7: 'KAMAZ_ALLKAMAZ_C',
                8: 'TOYOTA_RАV4_B',
                9: 'SCANIA_ALLSCANIA_C',
                10: 'RЕNАULТ_DUSТЕR_B'}
  image = np.asarray(Image.open(image_path).convert('RGB'))
  image_tensor = test_transform(image=image)['image']
  image_tensor = image_tensor.unsqueeze(0)

  model.eval()
  with torch.no_grad():
    logits = model(image_tensor).squeeze()
    pred = torch.argmax(logits).item()

  res = id2cartype[pred].split('_')
  return res[0], res[1], res[2]