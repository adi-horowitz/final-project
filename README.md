# SRM Network PyTorch
An implementation of "SRM : A Style-based Recalibration Module for Convolutional Neural Networks".

## Implementation
#### experiments.py
Run the requested model on the preffered dataset.
#### layer_blocks.py 
Implementation of SE block as well as SRM block. Including all variations and attemps for improving the original SRM block proposed in the paper.
#### resnet_with_block
Combine the different layer blocks implemented in layer_blocks.py with resnet architecture. 
#### Trainer
Containing a class abstracting the various tasks of training models, and a specific SRM trainer.
#### cifar10.py
An interface for loading cifar10 dataset.
#### plot.py print_fit_result.py
Those files are intended to print the results of the experiments.


## Training
#### python experiment.py --model_name <model_name>

model_name options are:
1. srmnet
2. senet
3. resnet
4. srm_with_corr
5. srm_with_median
6. srm_median_and_corr

```
## Training parameters
### Cifar
```python
batch_size = 128
epochs_count = 100
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                      weight_decay=1e-4)
scheduler = MultiStepLR(optimizer, [70, 80], 0.1)
```

## Results
### Cifar10
|           |ResNet32|Se-ResNet32|SRM-ResNet32|
|:----------|:-------|:----------|:-----------|
|accuracy   |92.1%   |92.5%      |89.7%       |





