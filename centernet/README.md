# CenterNet

[Objects as Points](https://arxiv.org/abs/1904.07850)

## 如何运行

以下为x86 Linux平台的运行过程，

### 准备

**python环境**

为了导出模型，您需要安装以下python包

+ mmengine >= 0.7.1
+ mmcv >= 2.0.0rc4
+ mmdet >= 3.0.0rc6

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

+ 可以从opencv-mobile的[Release](https://github.com/nihui/opencv-mobile/releases)页面选择一个版本下载预编译包，并相应修改CMakeLists.txt

**目录结构**

在使用opencv-mobile的情况下，当前工程应当有如下结构

```
detective
├── assets
├── centernet
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

模型的[结构](https://github.com/open-mmlab/mmdetection/blob/main/configs/centernet/centernet_r18_8xb16-crop512-140e_coco.py)和[权重](https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth)文件来自mmdetection的实现。

CenterNet使用最大池化代替NMS操作，原始实现中将池化操作放在解码中，这里参考CenterX的做法，将最大池化操作放在模型前向中。

1. 下载[权重](https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth)文件，并放到export目录下；
2. 导出onnx文件
```bash
cd detective/centernet/export
python export.py
```
3. 将onnx导出ncnn格式。可以直接在[这里](https://convertmodel.com/)进行转换
4. 将转换后的文件重命名为centernet-r18.param和centernet-r18.bin，并放到assets目录下

### 运行

```shell
cd detective/centernet
mkdir -p build
cd build/
cmake ..
make -j4
./centernet ../../assets/dog.jpg
```

## 感谢

+ [ncnn](https://github.com/Tencent/ncnn)
+ [opencv-mobile](https://github.com/nihui/opencv-mobile)
+ [mmdetection](https://github.com/open-mmlab/mmdetection)
+ [CenterX](https://github.com/JDAI-CV/centerX)
