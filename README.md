# GSCNN
This is the official code for:

#### Gated-SCNN: Gated Shape CNNs for Semantic Segmentation

[Towaki Takikawa](https://tovacinni.github.io), [David Acuna](http://www.cs.toronto.edu/~davidj/), [Varun Jampani](https://varunjampani.github.io), [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)

ICCV 2019
**[[Paper](https://arxiv.org/abs/1907.05740)]  [[Project Page](https://nv-tlabs.github.io/GSCNN/)]**

![GSCNN DEMO](docs/resources/gscnn.gif)

Based on based on https://github.com/NVIDIA/semantic-segmentation.

## License
```
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
## Usage

##### Clone this repo
```bash
git clone https://github.com/nv-tlabs/GSCNN
cd GSCNN
 ```

#### Python requirements 

Currently, the code supports Python 3
* numpy 
* PyTorch (<= 1.0.0 )
* scipy 
* scikit-imagea
* tensorboardX
* tqdm
* torch-encoding

#### Download pretrained models

Download the pretrained model from the [Google Drive Folder](https://drive.google.com/file/d/1wlhAXg-PfoUM-rFy2cksk43Ng3PpsK2c/view), and save it in 'checkpoints/'

#### Download inferred images

Download (if needed) the inferred images from the [Google Drive Folder](https://drive.google.com/file/d/105WYnpSagdlf5-ZlSKWkRVeq-MyKLYOV/view)

#### Evaluation (Cityscapes)
```bash
python train.py --evaluate --snapshot checkpoints/best_cityscapes_checkpoint.pth
```

If you use this code, please cite:

```
@article{takikawa2019gated,
  title={Gated-SCNN: Gated Shape CNNs for Semantic Segmentation},
  author={Takikawa, Towaki and Acuna, David and Jampani, Varun and Fidler, Sanja},
  journal={ICCV},
  year={2019}
}
```

