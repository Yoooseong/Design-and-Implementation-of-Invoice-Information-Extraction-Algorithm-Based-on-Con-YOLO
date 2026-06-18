# Design-and-Implementation-of-Invoice-Information-Extraction-Algorithm-Based-on-Con-YOLO
This algorithm integrates the contextual guidance capability into the native YOLOv11 framework through an attention mechanism. It utilizes the attention mechanism to explore the correlations between features of different scales, enhances the feature representation of small targets and complex scenes

## 🌟 项目简介

本项目旨在实现高效、精准的发票关键信息提取。通过在YOLOv11主干网络中引入**Context Guide模块**，增强了模型对上下文信息的捕捉能力，有效解决了发票场景中因褶皱、污损、光照不均导致的信息提取困难。

### 核心改进
- **算法架构**：基于 **YOLOv11**，引入 **Context Guide** 注意力机制。
- **特征增强**：设计了 **FGLo模块**，通过全局平均池化和全连接层动态调整通道权重，增强关键特征表达。
- **轻量化设计**：在保持高精度的同时，通过深度可分离卷积降低计算复杂度，单张发票处理仅需 **0.8秒**。

## 📊 实验结果与性能评估

在包含800张多种板式（含外卖、超市小票等）的自建数据集上进行训练与验证。实验结果表明，Con-YOLO在精确率和召回率上均优于原始YOLOv11及Tesseract-OCR。

### 核心指标对比

| 模型 | 精确率 (Precision) | 提升幅度 | 处理速度 |
| :--- | :---: | :---: | :---: |
| **Con-YOLO (本项目)** | **96.8%** | **+4.8%** | **0.8s / 张** |
| YOLOv11 (原版) | 92.0% | - | - |
| Tesseract-OCR | 83.2% | - | - |

### 详细评估指标

根据论文第4章实验分析，模型在验证集上的具体表现如下：

| 类别 (Class) | 精确率 (P) | 召回率 (R) | mAP@50 |
| :--- | :---: | :---: | :---: |
| **All (平均)** | **0.968** | **0.973** | **0.985** |
| Address (地址) | 0.980 | 0.993 | 0.989 |
| Date (日期) | 0.918 | 0.894 | 0.946 |
| Sum (总金额) | 0.965 | 0.953 | 0.966 |

### 训练结果可视化

下图展示了模型训练过程中的终端输出结果，包含最终的损失值（Loss）、精确率、召回率及mAP指标：

![终端训练输出结果](assets/OutputResult.png)
*(图注：模型训练最终指标输出，mAP@50达到0.985)*

---

### 曲线分析

#### 1. 精确率-召回率曲线 (PR Curve)
模型在各类别上均保持了较高的AP值，整体mAP@0.5达到0.985，证明模型在高置信度下能精准识别正样本。
*(此处可插入论文图4.6 PR曲线)*

#### 2. 损失函数下降曲线 (Loss Curve)
随着训练轮次增加，Box Loss和Cls Loss显著下降并趋于平稳，表明模型收敛性良好。
*(此处可插入论文图4.10 损失值曲线)*

## 🚀 快速开始

### 1. 环境依赖
请确保安装以下依赖库（建议使用Python 3.10+）：
```bash
pip install -r requirements.txt