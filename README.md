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
Copyright (C) 2019 NVIDIA Corporation. Towaki Takikawa, David Acuna, Varun Jampani, Sanja Fidler
All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Permission to use, copy, modify, and distribute this software and its documentation
for any non-commercial purpose is hereby granted without fee, provided that the above
copyright notice appear in all copies and that both that copyright notice and this
permission notice appear in supporting documentation, and that the name of the author
not be used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
~                                                                             
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
* PyTorch (>=1.1.0)
* torchvision
* scipy 
* scikit-image
* tensorboardX
* tqdm
* torch-encoding
* opencv
* PyYAML

#### Download pretrained models

Download the pretrained model from the [Google Drive Folder](https://drive.google.com/file/d/1wlhAXg-PfoUM-rFy2cksk43Ng3PpsK2c/view), and save it in 'checkpoints/'

#### Download inferred images

Download (if needed) the inferred images from the [Google Drive Folder](https://drive.google.com/file/d/105WYnpSagdlf5-ZlSKWkRVeq-MyKLYOV/view)

#### Evaluation (Cityscapes)
```bash
python train.py --evaluate --snapshot checkpoints/best_cityscapes_checkpoint.pth
```

#### Training

A note on training- we train on 8 NVIDIA GPUs, and as such, training will be an issue with WiderResNet38 if you try to train on a single GPU.

If you use this code, please cite:

```
@article{takikawa2019gated,
  title={Gated-SCNN: Gated Shape CNNs for Semantic Segmentation},
  author={Takikawa, Towaki and Acuna, David and Jampani, Varun and Fidler, Sanja},
  journal={ICCV},
  year={2019}
}
```

