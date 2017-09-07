Squeeze-and-Excitation Networks
---

This directory contains code to evaluate the classification models released by
the authors of the [paper](https://arxiv.org/abs/1709.01507):

```
Squeeze-and-Excitation Networks, 
Jie Hu, Li Shen, Gang Sun, arxiv 2017
```

This code is based on the [original](https://github.com/hujie-frank/SENet) 
implementation (which uses caffe).

### Pretrained Models

Each of the Squeeze-and-Excitation networks released by the authors has been imported into [MatConvNet](https://github.com/vlfeat/matconvnet) and can be downloaded here:

[SE Networks](http://www.robots.ox.ac.uk/~albanie/models.html#se-models)

The `run_se_benchmarks.m` script will evaluate each of these models on the ImageNet validation set. It will download the models automatically if you have not already done so (note that these evaluations require a copy of the imagenet data).  The accuracy of each pretrained model can also be found [here](http://www.robots.ox.ac.uk/~albanie/models.html#se-models). 

To give some idea of the relative computational burdens of model, esimates are provided below:


| model | input size | param memory | feature memory | flops |
|-------|------------|--------------|----------------|-------|
| [SE-ResNet-50](reports/SE-ResNet-50.md) | 224 x 224 | 107 MB | 103 MB | 4 GFLOPs|
| [SE-ResNet-101](reports/SE-ResNet-101.md) | 224 x 224 | 189 MB | 155 MB | 8 GFLOPs|
| [SE-ResNet-152](reports/SE-ResNet-152.md) | 224 x 224 | 255 MB | 220 MB | 11 GFLOPs|
| [SE-ResNeXt-50-32x4d](reports/SE-ResNeXt-50-32x4d.md) | 224 x 224 | 105 MB | 132 MB | 4 GFLOPs|
| [SE-ResNeXt-101-32x4d](reports/SE-ResNeXt-101-32x4d.md) | 224 x 224 | 187 MB | 197 MB | 8 GFLOPs|
| [SENet](reports/SENet.md) | 224 x 224 | 440 MB | 347 MB | 21 GFLOPs|
| [SE-BN-Inception](reports/SE-BN-Inception.md) | 224 x 224 | 46 MB | 43 MB | 2 GFLOPs|


Each estimate corresponds to computing a single element batch. This table was generated
with [convnet-burden](https://github.com/albanie/convnet-burden) - the repo has a list of the assumptions used produce estimations. Clicking on the model name should give a more detailed breakdown.


### Installation

The easiest way to use this module is to install it with the `vl_contrib` 
package manager:

```
vl_contrib('install', 'mcnSENets') ;
vl_contrib('setup', 'mcnSENets') ;
vl_contrib('test', 'mcnSENets') ; % optional
```

**Note:** The ordering of the imagenet labels differs from the standard ordering commonly found in caffe, pytorch etc.  These are remapped automically in the evaluation code.  The mapping between the synsets indices can be found [here](misc/label_map.txt).
