# RTMDet

[RTMDet: An Empirical Study of Designing Real-Time Object Detectors](https://arxiv.org/pdf/2212.07784v2.pdf)

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

**目录结构**

在使用opencv-mobile的情况下，当前工程应当有如下结构

```
detective
├── assets
├── rtmdet
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

模型的[结构](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py)和[权重](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth)文件来自mmyolo的实现。

1. 下载[权重](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth)文件，并放到export目录下；
2. 导出onnx文件
```bash
cd detective/rtmdet/export
python export.py
```
3. 将onnx导出ncnn格式。可以直接在[这里](https://convertmodel.com/)进行转换
4. 将转换后的文件重命名为rtmdet-tiny.param和rtmdet-tiny.bin，并放到assets目录下

### 运行

```shell
cd detective/rtmdet
mkdir -p build
cd build/
cmake ..
make -j4
./rtmdet ../../assets/dog.jpg
```

## 感谢

+ [ncnn](https://github.com/Tencent/ncnn)
+ [opencv-mobile](https://github.com/nihui/opencv-mobile)
+ [mmyolo](https://github.com/open-mmlab/mmyolo)
