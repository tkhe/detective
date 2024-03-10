# detective

支持多种常用目标检测算法在x86平台的ncnn部署

## 支持的算法

### **通用目标检测**

- [X] [CenterNet](centernet)
- [ ] RTMDet
- [ ] YOLOX
- [ ] YOLOv8
- [ ] Faster R-CNN

### **旋转目标检测**
- [ ] RTMDet-Rotate

### **人体姿态估计**
- [ ] YOLO-Pose

## 感谢

本项目的部署依赖以下项目

+ [ncnn](https://github.com/Tencent/ncnn)
+ [opencv-mobile](https://github.com/nihui/opencv-mobile)
+ [pnnx](https://github.com/pnnx/pnnx)

本项目的模型结构和权重文件来自以下项目

+ [detectron2](https://github.com/facebookresearch/detectron2)
+ [mmdetection](https://github.com/open-mmlab/mmdetection)
+ [mmyolo](https://github.com/open-mmlab/mmyolo)
