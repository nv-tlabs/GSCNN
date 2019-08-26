"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# Code adapted from:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py

Source License
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import six
import os.path as osp

from ast import literal_eval
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.nn import init


from utils.AttrDict import AttrDict


__C = AttrDict()
# Consumers can get config by:
# from fast_rcnn_config import cfg
cfg = __C
__C.EPOCH = 0
__C.CLASS_UNIFORM_PCT=0.0
__C.BATCH_WEIGHTING=False
__C.BORDER_WINDOW=1
__C.REDUCE_BORDER_EPOCH= -1
__C.STRICTBORDERCLASS= None

__C.DATASET =AttrDict()
__C.DATASET.CITYSCAPES_DIR='/home/username/data/cityscapes'
__C.DATASET.CV_SPLITS=3

__C.MODEL = AttrDict()
__C.MODEL.BN = 'regularnorm'
__C.MODEL.BNFUNC = torch.nn.BatchNorm2d
__C.MODEL.BIGMEMORY = False

def assert_and_infer_cfg(args, make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """

    if args.batch_weighting:
        __C.BATCH_WEIGHTING=True

    if args.syncbn:
        import encoding
        __C.MODEL.BN = 'syncnorm'
        __C.MODEL.BNFUNC = encoding.nn.BatchNorm2d
    else:
        __C.MODEL.BNFUNC = torch.nn.BatchNorm2d
        print('Using regular batch norm')

    if make_immutable:
        cfg.immutable(True)
