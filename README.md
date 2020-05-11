# Intro
This repo re-produced [paper](https://arxiv.org/abs/1807.11458) training codes.
I wrote this codes from [here](https://github.com/jingyang2017/Face-and-Image-super-resolution), where you can only get the test code, and implemented traing codes following the paper as much as I can. Because some hyper-parameters, such as the number of channels for each layer in High-to-Low generator and discriminators, were missing in the paper, they were set by my own choice.

In the paper, they used the spectral normalization. The spectral normalization code used in here is adopted from [repo](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py).


In this repo, all files with prefix `yoon_` are additionally impelmantatied files from the authors' [repo](https://github.com/jingyang2017/Face-and-Image-super-resolution).


```
Adrian Bulat, Jing Yang, Georgios Tzimiropoulos
"To learn image super-resolution, use a GAN to learn how to do image degradation first" in ECCV2018
```

## Requirements
```
Pytorch, Opencv, numpy, ...
```

## Data
* Trainset is in Dataset. HIGH is the training high resolution images. LOW is the training low resolution images 
* Testset is testset.tar
* locate the data folders using `High_Data` and `Low_Data` variables in `yoon_data.py`.

## Running testing
```
CUDA_VISIBLE_DEVICES=0, python model_evaluation.py 
```
## Fid Calculation
```
CUDA_VISIBLE_DEVICES=0, python fid_score.py /Dataset/HIGH/SRtrainset_2/ test_res/
```
This code is from https://github.com/mseitzer/pytorch-fid

## Running training
```
python yoon_train.py --gpu [your_gpu_num]
```
You can replace `[your_gpu_num]` with a number (gpu id).

If your dataset folders are different from the default setting, you have to locate the folders using `High_Data` and `Low_Data` variables in `yoon_data.py`.

## Re-produced results

![result1](./intermid_results/50_2_sr.png) ![result2](./intermid_results/50_3_sr.png) ![result3](./intermid_results/50_6_sr.png) ![result4](./intermid_results/50_7_sr.png) ![result5](./intermid_results/50_10_sr.png)

## License

This project is licensed under the MIT License
