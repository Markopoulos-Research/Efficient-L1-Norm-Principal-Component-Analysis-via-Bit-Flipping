## Efficient L1-norm Principal Component Analysis via bit-flipping (L1-PCA)

In this repo we implent the algorithm of [[1]](https://ieeexplore.ieee.org/document/7934025) for computing the L1-norm Principal-Component of a matrix. 
Formally, the provided script solves 

![equation](https://latex.codecogs.com/svg.image?\mathbf{Q}_{L1}&space;=&space;\underset{\mathbf{Q}&space;\in&space;\mathbb{R}^{D\times&space;K},&space;\mathbf{Q}^T\mathbf{Q}&space;=&space;\mathbf{I}_{K}&space;}{\rm&space;argmax}&space;||\mathbf{Q}^T&space;\mathbf{X}||_{1,1})

suboptimally with complexity O[NDmin{ND} + N^2 K^2 (K^2+d)].

---
* IEEEXplore article: https://ieeexplore.ieee.org/document/7934025
* Source code: https://github.com/RIT-MILOS-LAB/Efficient-L1-Norm-Principal-Component-Analysis-via-Bit-Flipping
---
**Questions/issues**

Inquiries regarding the scripts provided below are cordially welcome. In case you spot a bug, please let me know. If you use some piece of code for your own work, please cite the article above.

---
**Citing**

If you use our algorihthms, please cite [[1]](https://ieeexplore.ieee.org/document/7934025).

|[[1]](https://ieeexplore.ieee.org/document/7934025)|Markopoulos, P.P., Kundu, S., Chamadia, S. and Pados, D.A., 2017. Efficient L1-norm principal-component analysis via bit flipping. IEEE Transactions on Signal Processing, 65(16), pp.4252-4264.|
|-----|--------|
