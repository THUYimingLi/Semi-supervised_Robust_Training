
A repository that implements the semi-supervised robust training (SRT)  proposed in "Toward Adversarial Robustness via Semi-supervised Robust Training". [[arXiv]](https://arxiv.org/abs/2003.06974) 
This project is developed based on Python 3.6, created by [Yiming Li](http://liyiming.tech/) and [Yan Feng](http://yanfeng0096.com/). 


# Pixel-wise experiments
All code for this part are included in "pixel_exps" subfolder. Please change to that folder before running the code.
## Install prerequisites
```
pip install -r requirements.txt
```


## Running demos
### SRT training
* Train WideResNet-34-10 model on CIFAR-10 dataset

```
bash cifar_train.sh
```


* Train SmallCNN model on MNIST dataset

```
bash mnist_train.sh
```

### Robustness evaluation
* Evaluate robust WideResNet-34-10 model on CIFAR-10 by PGD attack

```
python cifar_eval.py 
```

* Evaluate robust SmallCNN model on MNIST by PGD attack

```
python mnist_eval.py 
```

## Download pre-trained model
Download the folder "checkpoints" [[download link]](https://www.dropbox.com/sh/9ec1s7nlrkeplwn/AADnNNeHmSip4lEhZJs0L1BRa/checkpoints?dl=0&subfolder_nav_tracking=1) and put it within the "pixel_exps" folder, then you can conduct
the robustness evaluation without training.






# Spatial experiments
## Install prerequisites
```
pip install -r requirements.txt
```


## Running demos

### SRT training
* Train ResNet model on CIFAR-10 dataset

```
bash CIFAR_train.sh
```


* Train SmallCNN model on MNIST dataset

```
bash MNIST_train.sh
```

### Robustness evaluation
* Evaluate robust ResNet model on CIFAR-10 by GridAdv attack

```
bash CIFAR_eval.sh
```

* Evaluate robust SmallCNN model on MNIST by GridAdv attack

```
bash MNIST_eval.sh
```

## Download pre-trained model
Download the  "checkpoints" [[download link]](https://www.dropbox.com/sh/3bbf00w0ykdxgpe/AAAEyYWVm70qbRTZD3BkjRela/checkpoints?dl=0&subfolder_nav_tracking=1) and put it within the "spatial_exps" folder, then you can conduct
the robustness evaluation without training.
