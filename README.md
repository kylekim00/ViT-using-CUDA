# FlashAttention Implementation using CUDA
To run the example code, ```nvcc flashAttention_Tensor.cu MHA.cu Easy_Tensor/easy_tensor_struct.cu ./nn.cu```.

<hr/>

## Description

- ```./Easy_Tensor/```: Easy Tensor, Tools for cuda in C, submodule <br/>
- ```./PPT/```: PResentation image description <br/>
- ```./Pre_weights/```: pretrained weights of ```timm``` module<br/>
- ```./MHA.cu```: Multiheaded Attention function, flashAttention implemented<br/>
- ```./MHA.h```: header file for MHA.cu<br/>
- ```./ViT_pretrained.ipynb```: ```timm``` module pre-trained weight binary file extraction<br/>
- ```./flashAttention_Tensor.cu```: main file of project<br/>
- ```./prop.cu```: device check<br/>


![alt text](https://github.com/kylekim00/Neural-Network-using-CUDA/blob/main/PPT/1.JPG?raw=true)
![alt text](https://github.com/kylekim00/Neural-Network-using-CUDA/blob/main/PPT/2.JPG?raw=true)
![alt text](https://github.com/kylekim00/Neural-Network-using-CUDA/blob/main/PPT/3.JPG?raw=true)
![alt text](https://github.com/kylekim00/Neural-Network-using-CUDA/blob/main/PPT/4.JPG?raw=true)
![alt text](https://github.com/kylekim00/Neural-Network-using-CUDA/blob/main/PPT/5.JPG?raw=true)
![alt text](https://github.com/kylekim00/Neural-Network-using-CUDA/blob/main/PPT/6.JPG?raw=true)
![alt text](https://github.com/kylekim00/Neural-Network-using-CUDA/blob/main/PPT/7.JPG?raw=true)
.
