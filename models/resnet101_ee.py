#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/models/resnet.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# PyTorch implementation of the ResNet backbone for use as a feature extractor
# in Faster R-CNN. See the Backbone base class for a description of how the
# classes here are structured.
#
# References
# ----------
# [1] "Deep Residual Learning for Image Recognition"
#     Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#

# from enum import Enums
import sys
import os

from math import ceil
import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from ..datasets import image
from .backbone import Backbone
from .myResNet import ResNet_2EE, BottleNeck,  get_output_shape, resnet101
from torchsummary import summary
device = 'cuda'

# class Architecture(Enum):
#   ResNet50 = "ResNet50"
#   ResNet101 = "ResNet101"
#   ResNet152 = "ResNet152"

class ConvBasic(nn.Module):
    def __init__(self, chanIn, chanOut, k=3, s=1,
                 p=1, p_ceil_mode=False, bias=False):
      super(ConvBasic, self).__init__()
      self.conv = nn.Sequential(
        nn.Conv2d(chanIn, chanOut, kernel_size=k, stride=s,
                  padding=p, bias=bias),
        nn.BatchNorm2d(chanOut),
        nn.ReLU(True)  # in place
      )

    def forward(self, x):
      return self.conv(x)

class IntrFE(nn.Module):
    # intermediate classifer head to be attached along the backbone
    # Inpsired by MSDNet classifiers (from HAPI):
    # https://github.com/kalviny/MSDNet-PyTorch/blob/master/models/msdnet.py

    def __init__(self, chanIn, interChans=64, last_conv_chan=1024):
      super(IntrFE, self).__init__()

      # index for the position in the backbone layer
      # input shape to automatically size linear layer

      # intermediate conv channels
      # interChans = 128 # TODO reduce size for smaller nets
      self.interChans = interChans
      self.last_conv_chan = last_conv_chan
      # conv, bnorm, relu 1
      layers = nn.ModuleList()
      self.conv1 = ConvBasic(chanIn, interChans, k=5, s=2, p=2)
      layers.append(self.conv1)
      self.conv2 = ConvBasic(interChans, interChans, k=3, s=2, p=1)
      layers.append(self.conv2)
      self.conv3 = ConvBasic(interChans, last_conv_chan, k=3, s=1, p=1)
      layers.append(self.conv3)
      self.layers = layers

    def forward(self, x):
      for layer in self.layers:
        x = layer(x)
      return x

class FeatureExtractor(nn.Module):
  def __init__(self, resnet):
    super().__init__()
    self._feature_extractor1 = nn.Sequential(
      resnet.conv1,     # 0
      resnet.bn1,       # 1
      resnet.relu,      # 2
      resnet.maxpool   # 3
    )
    self._feature_extractor2 = nn.Sequential(
      resnet.layer1,    # 4
      resnet.layer2,    # 5
      # resnet.layer3     # 6
    )

    # create exit by FE(Feature Extractor)
    self.exits = nn.ModuleList()
    self.exit_num = 2
    self.exit_loss_weights = [1.0, 1.0]  # for training need to match total exits_num
    self.exit_threshold = t.tensor([0.8],
                                       dtype=t.float32).to(device)  # for fast inference  #TODO: inference variable(not constant 0.8) need to make parameter
    self._build_exits(resnet)

    # Freeze initial layers
    self._freeze(resnet.conv1)
    self._freeze(resnet.bn1)
    self._freeze(resnet.layer1)

    # Ensure that all batchnorm layers are frozen, as described in Appendix A
    # of [1]
    self._freeze_batchnorm(self._feature_extractor1)
    self._freeze_batchnorm(self._feature_extractor2)
    self._freeze_batchnorm(self.exits[0])
    self._freeze_batchnorm(self.exits[1])

  # Override nn.Module.train()
  def train(self, mode = True):
    super().train(mode)

    #
    # During training, set all frozen blocks to evaluation mode and ensure that
    # all the batchnorm layers are also in evaluation mode. This is extremely
    # important and neglecting to do this will result in severely degraded
    # training performance.
    #
    if mode:
      # Set fixed blocks to be in eval mode
      if mode:
        # Set fixed blocks to be in eval mode
        self._feature_extractor1.eval()
        self._feature_extractor2.eval()
        self._feature_extractor2[1].train()
        self.exits[1].train()

        # *All* batchnorm layers in eval mode
        def set_bn_eval(module):
          if type(module) == nn.BatchNorm2d:
            module.eval()

        self._feature_extractor1.apply(set_bn_eval)
        self._feature_extractor2.apply(set_bn_eval)


  def forward(self, image_data):
    res = []

    y = self._feature_extractor1(image_data)
    res.append(self.exits[0](y))

    y = self._feature_extractor2(y)
    res.append(self.exits[1](y))
    # print(res[0])
    # print(res[1])
    return res[0], res[1]


  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)

  def _build_exits(self, resnet):
    # input shape ? does it have to be fixxed?
    ee1 = IntrFE(64)
    self.exits.append(ee1)

    # final exit
    self.exits.append(resnet.layer3)


class PoolToFeatureVector(nn.Module):
  def __init__(self, resnet):
    super().__init__()
    self._layer4 = resnet.layer4
    self._freeze_batchnorm(self._layer4)

  def train(self, mode = True):
    # See comments in FeatureVector.train()
    super().train(mode)
    if mode:
      def set_bn_eval(module):
        if type(module) == nn.BatchNorm2d:
          module.eval()
      self._layer4.apply(set_bn_eval)

  def forward(self, rois):
    y = self._layer4(rois)  # (N, 1024, 7, 7) -> (N, 2048, 4, 4)

    # Average together the last two dimensions to remove them -> (N, 2048).
    # It is also possible to max pool, e.g.:
    # y = F.adaptive_max_pool2d(y, output_size = 1).squeeze()
    # This may even be better (74.96% mAP for ResNet50 vs. 73.2% using the
    # current method).
    y = y.mean(-1).mean(-1) # use mean to remove last two dimensions -> (N, 2048)
    return y

  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)


class ResNetBackbone(Backbone):
  def __init__(self, architecture):
    super().__init__()

    # Backbone properties. Image preprocessing parameters are common to all
    # Torchvision ResNet models and are described in the documentation, e.g.,
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
    self.feature_map_channels = 1024  # feature extractor output channels
    self.feature_pixels = 16          # ResNet feature maps are 1/16th of the original image size, similar to VGG-16 feature extractor
    self.feature_vector_size = 2048   # linear feature vector size after pooling
    self.image_preprocessing_params = image.PreprocessingParams(channel_order = image.ChannelOrder.RGB, scaling = 1.0 / 255.0, means = [ 0.485, 0.456, 0.406 ], stds = [ 0.229, 0.224, 0.225 ])

    # Construct model and pre-load with ImageNet weights
    if architecture == "ResNet101":
      script_dir = os.path.dirname(__file__)
      pth_rel_path = "pretrained_model"
      file_name = "pretrained_resnet.pth"
      abs_file_path = os.path.join(script_dir, pth_rel_path)
      try:
        if not os.path.exists(abs_file_path):
          os.makedirs(abs_file_path)
      except OSError:
        print('Error: Creating directory. ' + abs_file_path)
      abs_file_path = os.path.join(abs_file_path, file_name)
      resnet_a = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
      t.save(resnet_a.state_dict(), abs_file_path)
      resnet = resnet101().to('cuda')
      resnet.load_state_dict(t.load(abs_file_path))
      # summary(resnet, (3, 32, 32), device='cuda')

    else:
      raise ValueError("Invalid ResNet architecture value: %s" % architecture.value)
    # print("Loaded IMAGENET1K_V1 pre-trained weights for Torchvision %s backbone" % architecture.value)

    # Feature extractor: given image data of shape (batch_size, channels, height, width),
    # produces a feature map of shape (batch_size, 1024, ceil(height/16), ceil(width/16))
    self.feature_extractor = FeatureExtractor(resnet = resnet)

    # Conversion of pooled features to head input
    self.pool_to_feature_vector = PoolToFeatureVector(resnet = resnet)

  def compute_feature_map_shape(self, image_shape):
    """
    Computes feature map shape given input image shape. Unlike VGG-16, ResNet
    convolutional layers use padding and the resultant dimensions are therefore
    not simply an integral division by 16. The calculation here works well
    enough but it is not guaranteed that the simple conversion of feature map
    coordinates to input image pixel coordinates in anchors.py is absolutely
    correct.

    Parameters
    ----------
    image_shape : Tuple[int, int, int]
      Shape of the input image, (channels, height, width). Only the last two
      dimensions are relevant, allowing image_shape to be either the shape
      of a single image or the entire batch.

    Returns
    -------
    Tuple[int, int, int]
      Shape of the feature map produced by the feature extractor,
      (feature_map_channels, feature_map_height, feature_map_width).
    """
    image_width = image_shape[-1]
    image_height = image_shape[-2]
    return (self.feature_map_channels, ceil(image_height / self.feature_pixels), ceil(image_width / self.feature_pixels))