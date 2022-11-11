# Decoupled and Boosted Learning for Skeleton-based Dynamic Hand Gesture Recognition

## Introduction
In this paper, we propose a lightweight dual-stream framework based on decoupled and boosted learning for skeleton-based dynamic hand gesture recognition. We evaluate our model on three challenging datasets: SHREC’17 Track dataset, FPHA dataset, and DHG-14/28 dataset. Experimental results show the superiority of our method.

## Requirements

* Python 3.8
* Tensorflow 2.4.1
* numpy
* tqdm
* scipy
* opencv

## Downlaod Dataset
*  You need to execute following operations to process datasets.
1. Download the [SHREC’17 Track Dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/), [FPHA Dataset](https://guiggh.github.io/publications/first-person-hands/) or [DHG-14/28 Dataset](http://www-rech.telecom-lille.fr/DHGdataset/).
2. Set the path to your downloaded dataset folder in the ```datasets/SHREC.py``` or ```datasets/FPHA.py``` or ```datasets/DHG.py```.
3. Run the following commands to process datasets.
```
python datasets/FPHA.py        # on FPHA Dataset
python datasets/SHREC.py       # on SHREC’17 Track Dataset
python datasets/DHG.py         # on DHG-14/28 Dataset
```

## Test Model
* You can directly download the trained models from [here](https://drive.google.com/drive/folders/1oigD381_oiKLMkajePgzP2TTgJN9AmCS?usp=sharing).
1. Run the following command to test models.
```
python test.py
```
