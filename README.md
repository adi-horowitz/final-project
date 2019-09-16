# SRM Network PyTorch
An implementation of SRM block, proposed in "SRM : A Style-based Recalibration Module for Convolutional Neural Networks".

## Training
python experiment.py --model_name <model_name>

while model_name is one of:
1. srmnet
2. se
3. resnet
4. srm_with_corr
5. srm_with_median
6. srm_with_median_and_corr_matrix


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





