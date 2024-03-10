# YOLOX

[YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)

## 如何运行

以下为x86 Linux平台的运行过程

### 准备

**python环境**

为了导出模型，您需要安装以下python包

+ mmengine >= 0.7.1
+ mmcv >= 2.0.0rc4
+ mmdet >= 3.0.0rc6
+ mmyolo >= 0.3.0

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
├── yolox
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

模型的[结构](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolox/yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco.py)和[权重](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco/yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco_20230210_143637-4c338102.pth)文件来自mmyolo的实现。

YOLOX的模型中含有Focus层，如果使用pytorch->onnx->ncnn的导出方式，得到的onnx算子比较碎，且需要手工修改ncnn的param文件，比较麻烦，这里我们采用pnnx的导出方式。关于pnnx的详细介绍可参考[pnnx](https://github.com/pnnx/pnnx)。另外，mmdet中Focus实际包含了Focus和Conv两个算子，这里我们将其拆开，使Focus不需要导出权重

1. 下载[权重](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco/yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco_20230210_143637-4c338102.pth)文件，并放到export目录下；
2. 导出pt文件：
```bash
cd detective/yolox/export
python export.py
```
3. 得到model.pt后，导出ncnn
```shell
./pnnx model.pt \
    inputshape=[1,3,640,640] \
    inputshape2=[1,3,416,416] \
    moduleop=mmdet.models.backbones.csp_darknet.Focus
```
4. 将转换后的model.ncnn.param和model.ncnn.bin重命名为yolox-tiny.param和yolox-tiny.bin，并放到assets目录下

### 运行

```shell
cd detective/yolox
mkdir -p build
cd build/
cmake ..
make -j4
./yolox ../../assets/dog.jpg
```

## 感谢

+ [ncnn](https://github.com/Tencent/ncnn)
+ [opencv-mobile](https://github.com/nihui/opencv-mobile)
+ [mmyolo](https://github.com/open-mmlab/mmyolo)
