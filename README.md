# CIFAR-10

I running a custom Pytorch 14-layer ResNet neural network to classify CIFAR-10 dataset.

The dataset consists of 50 000 training and 10 000 testing images with a size of 32*32. Every image belongs to one of ten classes.
```
airplane automobile bird cat deer dog frog horse ship truck
```

![](CIFAR-10.jpg)

The strategy is based on pooling steps to decrease width and height dimensions to 1*1 and then pass it to feed-forward classifier.

I played some time with custom hyperparameter tuning and got 92.07% accuracy on the test dataset.

I didn't try other architectures and a number of layers and decided to look at results that would after automatic(optuna package) hyperparameter tuning.
