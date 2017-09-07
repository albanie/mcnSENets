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

Each of the Squeeze-and-Excitation networks released by the authors has been imported into MatConvNet and can be downloaded here:

[SE Networks](http://www.robots.ox.ac.uk/~albanie/models.html#se-models)

The `run_se_benchmarks.m` script will evaluate each of these models on the ImageNet validation set. It will download the models automatically if you have not already done so (note that these evaluations require a copy of the imagenet data).  

The result of this evaluation for each pretrained model can be found [here](http://www.robots.ox.ac.uk/~albanie/models.html#se-models). 


### Installation

The easiest way to use this module is to install it with the `vl_contrib` 
package manager:

```
vl_contrib('install', 'mcnSENets') ;
vl_contrib('setup', 'mcnSENets') ;
```

**Note:** The ordering of the imagenet labels differs from the standard ordering commonly found in caffe, pytorch etc.  These are remapped automically in the evaluation code.  The mapping between the synsets indices can be found [here](misc/label_map.txt).