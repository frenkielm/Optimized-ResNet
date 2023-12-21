# Optimized-ResNet

[code](https://github.com/frenkielm/Optimized-ResNet)

## 1. Instruction

1. (baseline)Repetition the ResNet on Cifar-10 from chapter 4 of the paper  [**Deep Residual Learning for Image Recognition**](https://arxiv.org/abs/1512.03385) with a little change about optimizer.
2. Try advanced data augmentation such as Mixup on the baseline model and test whether could it decline the classification errors of the model on Cifar-10 test set. 

## 2. Hyperparameters settings

* learning_rate: 0.001
* weight_decay : 5e<sup>-5</sup>
* epoch: 200
* optimizer: Adam
* lr_dacay: decline by 10 if the model does not update the validation set loss value within 10 consecutive epochs
* data augmentation:
  1. baseline: RandomHorizontalFlip, RandomCrop with padding equal to 4, Cutout
  2. Mixup: Mixup and the augmentation in baseline

## 3. Classification errors on Cifar-10 test set

| Model     | baseline | Mixup |
| --------- | -------- | ----- |
| **Res20** | 8.98     |       |
| **Res32** | 8.30     |       |
| **Res44** |          |       |
| **Res56** |          |       |

