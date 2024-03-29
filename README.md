# SeisGAN: Improving Seismic Image Resolution and Reducing Noise Using a Generative Adversarial Network
An application of generative adversarial networks to seismic data processing (resolution ehancement and denoising). This is a repository for the paper "SeisGAN: Improving Seismic Image Resolution and Reducing Noise Using a Generative Adversarial Network".


## Example
![image](https://github.com/leilin1995/Higher-Resolution-and-Less-Noisy-Seismic-Images-An-Application-of-Generative-Adversarial-Neural-Net/blob/master/application/k3/crossline.png)

## Project Organization

![image](https://github.com/leilin1995/SeisGAN-Improving-Seismic-Image-Resolution-and-Reducing-Noise/blob/master/Organization.png)


## Code

All training and test code are in the directory **code**.

## Dataset

The three filed seismic images and the reconstructed images by our method are in **application** folder.The synthetic seismic data used for training can be obtained by visting the "https://www.kaggle.com/datasets/leilin1995/seisgan".

## Dependencies

* python 3.6.13
* pytorch 1.9.1
* torchvision 0.10.1
* tqdm 4.62.3
* scipy 1.5.4
* numpy 1.19.5
* h5py 3.1.0
* pandas 1.1.5
* PIL 8.4.0
* matplotlib 3.3.4

## Usage instructions
Download this project and build the dependency.
Then use application_filed.py --test_data_path="your data path" --save_path="your save path of resed by GAN"

## Citation

If you find this work useful in your research, please consider citing:

```
Lin L, Zhong Z, Cai C, et al. SeisGAN: Improving Seismic Image Resolution and Reducing Random Noise Using a Generative Adversarial Network[J]. Mathematical Geosciences, 2023: 1-27.
```

BibTex

```
@article{lin2023seisgan,
  title={SeisGAN: Improving Seismic Image Resolution and Reducing Random Noise Using a Generative Adversarial Network},
  author={Lin, Lei and Zhong, Zhi and Cai, Chuyang and Li, Chenglong and Zhang, Heng},
  journal={Mathematical Geosciences},
  pages={1--27},
  year={2023},
  publisher={Springer}
}

```
