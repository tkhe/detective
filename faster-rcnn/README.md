# Faster R-CNN

[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

## 如何运行

以下为x86 Linux平台的运行过程

### 准备

**python环境**

为了导出模型，您需要安装以下python包

+ detectron2 >= 0.4.1

**ncnn**

可以从ncnn的[Release](https://github.com/Tencent/ncnn/releases)页面直接下载预编译包，或者按照ncnn的[wiki](https://github.com/Tencent/ncnn/wiki/how-to-build)从源码进行编译安装

如果您选择预编译包，解压缩后将ncnn-xxxx-xx目录移动到detective目录下，并重命名为ncnn

如果您从源码编译，将编译后得到的install目录移动到detective目录下，并重命名为ncnn

**opencv-mobile**

+ 如果您已经安装了opencv，可以选择跳过这一步，并相应地修改CMakeLists.txt，使其能够链接opencv

+ 如果您是从源码编译ncnn，并且开启了NCNN_SIMPLEOCV选项（如下所示），同样可以跳过这一步，删除CMakeLists.txt中OpenCV的部分

```cmake
option(NCNN_SIMPLEOCV "minimal opencv structure emulation" ON)
```

+ 可以从opencv-mobile的[Release](https://github.com/nihui/opencv-mobile/releases)页面选择一个版本下载预编译包，解压缩后移动到detective目录下，并重命名为opencv-mobile

**pnnx**

可以从pnnx的[Release](https://github.com/pnnx/pnnx/releases)页面直接下载预编译包

**目录结构**

在使用opencv-mobile的情况下，当前工程应当有如下结构

```
detective
├── assets
├── faster-rcnn
├── ncnn
│   ├── bin
│   ├── include
│   └── lib
├── opencv-mobile
│   ├── bin
│   ├── include
│   ├── lib
│   └── share
├── ...
├── LICENSE
├── README.md
```

### 导出

Faster R-CNN有多种变体，如C4、FPN、DilatedC5等，它们之间的差异可以参考[这里](https://github.com/facebookresearch/detectron2/tree/main/configs)，我们选择最常用的FPN结构进行部署

模型的[结构](https://github.com/facebookresearch/detectron2/blob/main/configs/common/models/mask_rcnn_fpn.py)和[权重](https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl)文件来自detectron2的实现。

detectron2有yaml和LazyConfig两种配置文件，这里我们使用LazyConfig配置文件，但是detectron2并没有提供Faster R-CNN的配置文件，因此我们在Mask R-CNN配置文件的基础上删除mask_head的相关配置

1. 下载[权重](https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl)文件，并放到export目录下；
2. 导出pt文件：
```bash
cd detective/faster_rcnn/export
python export.py
```
3. 将rpn.pt导出ncnn
```shell
./pnnx rpn.pt \
    inputshape=[1,3,640,640] \
    inputshape2=[1,3,416,416]
```
4. 将转换后的model.ncnn.param和model.ncnn.bin重命名为rpn.param和rpn.bin，并放到assets目录下
5. 将rcnn.pt导出ncnn
```
./pnnx rcnn.pt inputshape=[1,256,7,7]
```
6. 将转换后的model.ncnn.param和model.ncnn.bin重命名为rcnn.param和rcnn.bin，并放到assets目录下

**注意**

从网络结构上可以看到RPN和RCNN两部分都不含有特殊的算子，因此实际上也可以通过onnx的导出方式进行导出，只需在cpp文件中修改成相应的输入和输出层的名字即可

### 运行

```shell
cd detective/faster_rcnn
mkdir -p build
cd build/
cmake ..
make -j4
./faster_rcnn ../../assets/dog.jpg
```

## 感谢

+ [ncnn](https://github.com/Tencent/ncnn)
+ [opencv-mobile](https://github.com/nihui/opencv-mobile)
+ [detectron2](https://github.com/facebookresearch/detectron2)
