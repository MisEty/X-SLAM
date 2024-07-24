# X-SLAM: Scalable Dense SLAM for Task-aware Optimization using CSFD
Zhexi Peng, Yin Yang, Tianjia Shao, Chenfanfu Jiang, Kun Zhou
![Teaser image](assets/teaser.png)
This repository contains the official authors implementation associated with the paper "X-SLAM: Scalable Dense SLAM for Task-aware Optimization using CSFD", which can be found [here](https://gapszju.github.io/X-SLAM/static/pdfs/X-SLAM_final.pdf).

Abstract: *We present X-SLAM, a real-time dense differentiable SLAM system that leverages the complex-step finite difference (CSFD) method for efficient calculation of numerical derivatives, bypassing the need for a large-scale computational graph. The key to our approach is treating the SLAM process as a differentiable function, enabling the calculation of the derivatives of important SLAM parameters through Taylor series expansion within the complex domain. Our system allows for the real-time calculation of not just the gradient, but also higher-order differentiation. This facilitates the use of high-order optimizers to achieve better accuracy and faster convergence. Building on X-SLAM, we implemented end-to-end optimization frameworks for two important tasks: camera relocalization in wide outdoor scenes and active robotic scanning in complex indoor environments. Comprehensive evaluations on public benchmarks and intricate real scenes underscore the improvements in the accuracy of camera relocalization and the efficiency of robotic navigation achieved through our task-aware optimization.*

## 1. Code Structure
This repository consists of the following parts:
### DeviceArray, Visualization, XKinectFusion, Common
Here are the main algorithm implementations from the paper. `DeviceArray` is our implementation of a mathematical operations library based on second-order complex numbers (C++ & CUDA). `Common` includes auxiliary tools such as timing and memory management (referenced from *Programming in Parallel with CUDA*). `Visualization` is our visualization tool (not yet open-sourced.) and some 3D data structures. `XKinectFusion` is the differentiable KinectFusion system based on CSFD.
### Experiments
Here are some demos to show our method:\
`test_xkinect_fusion`: A simple example of running XKinectFusion. It will output FPS, camera trajectory, and reconstructed point cloud.\
`test_CSFD`: A simple example of using CSFD to compute numerical differentiation. It will show the performance of our acceleration method for CSFD and how to use DCSFD to compute high-order differentiation.

## 2. Installation
### 2.1 Clone the Repository
```
git clone https://github.com/MisEty/X-SLAM
```
### 2.2 Install dependencies
Our code relies on the following third-party libraries:
```
Sophus
yaml-cpp
opencv
Eigen 
CUDAToolkit
```
Our code has been tested on Ubuntu 22.04 + CUDA 11.8 + NVIDA GeForce RTX 4090 + Intel Core i9-13900KF. It should also run on comparable platforms

### 2.3 Install Source Code
Move to project directory, and type the following command:
```
mkdir build
cd build
cmake ..
make
```
Then the code could be compiled.

## 3. X-KinectFusion
### 3.1 Prepare datasets
Currently, we provide support for reading the 7-Scenes and ICL-NIUM datasets. Please download the datasets yourself and place them in the appropriate locations.
### 3.2 Run
The runtime parameters are set through a YAML file. You can refer to our example (`Experiments/test_xkinect_fusion/configs/ICL_traj2.yaml`). Then type 
```
./Experiments/test_xkinect_fusion/test_kinect_fusion your_config.yaml
```
in build foler to run.

## 4. Todo
- [ ] Camera relocalization demo on 7-Scenes dataset
- [ ] Robot active scanning demo on virtual scenes

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{peng2024xslam,
        author    = {Zhexi Peng and Yin Yang and Tianjia Shao and Chenfanfu Jiang and Kun Zhou},
        title     = {X-SLAM: Scalable Dense SLAM for Task-aware Optimization using CSFD},
        booktitle  = {ACM SIGGRAPH Conference Proceedings, Denver, CO, United States, July 28 - August 1, 2024},
        year      = {2024},
      }</code></pre>
  </div>
</section>