---
layout: post
title: Swin Transformer
data: 2021-12-16
excerpt: "Swin Transformer"
tags: [transformer]
coments: false
mathjax: true
---

# Swin Transformer

- 제목: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
- 저자: Ze Liu 외 7명, Microsoft Research Asia
- 인용수: 495회
- 학술지: ICCV 2021

## Introduction

- vision 계열에서는 CNN, NLP 계열에서는 transformer가 대부분 차지하고 있음
- 본 논문에서는 transformer 구조에 object detection의 성질을 반영할 수 있는 방법을 제안하며, vision뿐만 아니라 NLP에서도 사용이 가능한 backbone을 제안함
- transformer를 vision에 적용시키고자 많은 방법이 제안되었지만, 문제점들이 존재함
  1. 다양한 scale을 사용하지 못함
  2. 해상도가 높을수록, scale이 증가할수록 연산량이 급격하게 증가함
  3. 기존의 transformer base model은 classification을 위한 모델로 제안됨

- 이를 해결하기 위해 Shifted Window기법을 사용한 Swin Transformer를 제안함

![](C:\workspace\논문\swin_transformer\fig1.png)

- image 안에 local window가 존재
- local window안에서 self attention을 수행하며 layer를 통과할 수록 인접한 patch부분을 합쳐 계층적으로 표현할 수 있게 됨

- local window 경계에 있는 patch는 self attention을 수행하지 않기 때문에 window 크기의 절반만큼 shift시킨 후 같은 계산을 반복해서 수행함

![](C:\workspace\논문\swin_transformer\fig2.png)

- shifted window방식을 사용하여 image classification, object detection, semantic segmentation에서 ViT, DeiT, RexNe(X)t를 능가함



## Method

- 전체적인 architecture (Tiny 모델)

![](C:\workspace\논문\swin_transformer\fig3.png)

- input ~ patch partition
  - input으로 image를 받는데, patch partition에서 이미지를 patch로 나누게 됨
  - 나눠진 patch는 NLP에서의 token과 같이 사용되며, patch 하나의 feature는 RGB 채널을 이어붙인(concatenate) 형태
  - fig에 사용된 patch는 4\*4를 사용하여 한 feature는 4\*4\*3=48

- Stage 1

  - Linear Embedding에서 h\*w\*48을 h\*w\*c로 바꾸어 줌

  - Swin Transformer Block은 (b)와 같이 구성되어 있음

  - 일반적인 MSA(Multi-head Self Attention)대신 W-MSA(Window-MSA)와 SW-MSA(Shifted Window MSA)를 사용

    > W-MSA: Local Window안에서(patch)의 self attention
    >
    > SW-MSA: Local Window끼리 self attention

- Stage 2~4

  - Patch Merging에서 계층구조의 feature map을 생성하기위하여 인접한 patch를 결합하여 하나의 patch를 생성

  - 인접한 2\*2의 patch를 concatenate하는데, 이러면 채널은 4C가 되기 때문에 linear layer를 통과시켜 2C로 조정해줌

    > 2C로 조정해주는 이유?
    >
    > 4C로 그대로 쓰면 stage를 지날때마다 채널수가 너무 많이 늘어나 속도가 느려질수있다?

  - 이를 통해 해상도는 줄어들며 채널은 늘어나게 됨

-  기존 attention 모델은 이미지의 모든 픽셀의 쌍에 대해 연산하기 때문에 연산량이 매우 급격하게 늘어남
- 이를 shifted window를 통해 해결
  - 하나의 patch는 4\*4의 pixel로 구성
  - 하나의 Window는 M\*M의 patch로 구성, 본 논문에서 M=7을 사용
  - 하나의 이미지에는 h\*w개의 Window가 존재
- 하나의 patch 안에서만 계산되기 때문에 M\*M에 따라 연산량이 늘어나게 되어 이미지 전체에 비해 훨씬 적은 연산량을 갖게됨

### Shifted window partitioning in successive blocks

- 인접한 patch들끼리 attention을 할 수 없는 문제가 있음
- 이를 위해 Window를 M/2만큼 Cyclic shift한 다음 같은 계산을 수행

![](C:\workspace\논문\swin_transformer\fig4.png)

- 원래는 4\*4크기의 window를 갖고있는데, window 끼리 attention을 적용시키기 위해 window를 shift하니 window의 크기가 매우 다양해짐

- 이를 통일시켜주기위해 원래 있던 patch를 짤린 부분에 붙여주며 mask를 씌워 attention에서는 제외를 시켜줌

- attention을 진행한 후에는 다시 원래의 형태로 되돌림(reverse cyclic shift)

- 이를 통해 window끼리의 연결성을 확보함

  > padding을 사용할 수도 있지만 연산량 증가하기 때문에 사용하지 않음

### Relative position bias

- VIT와 다르게 처음에 더해주는 position embedding이 없음
- 대신 attention과정에서 relative position bias(B)를 추가해 주었음

![](C:\workspace\논문\swin_transformer\s4.png)

- VIT에서는 절대 좌표를 넣어주었음

- 본 논문에서는 절대 좌표보다는 상대 좌표가 더 좋은 방법이라며 제시함

  - (0, 0)픽셀에서 (3, 3)으로 가기위해서는 (3, 3)만큼 이동해야함
  - 반대로 (3, 3)에서 (0, 0)으로 가기위해서는 (-3, -3)만큼 이동해야함
  - 즉, 어떠한 픽셀을 중심으로 이동해야하는 좌표값이 달라지는것을 말함

  

## Architecture Variants

- Swin-T(Tiny): C=96, layer numbers={2, 2, 6, 2}
- Swin-S(Small): C=96, layer numbers={2, 2, 18, 2}
- Swin-B(Base): C=128, layer numbers={2, 2, 18, 2}
- Swin-L(Large): C=192, layer numbers={2, 2, 18, 2}



## Experiments

- Use Dataset
  - Classification: ImageNet
  - Object Detection: COCO
  - Sementic Segmentation: ADE20K

#### Classification

![](C:\workspace\논문\swin_transformer\table1.png)

- (a): ImageNet-1K만을 학습시킨 결과
- (b): ImageNet-22K로 사전학습 후 ImageNet-1K로 미세조정한 결과
- setting
  - optimizer: AdamW
  - initial learning rate: 0.001
  - batch size: 1024
  - GPU: V100

### Object Detection

![](C:\workspace\논문\swin_transformer\table2.png)

setting

- optimizer: AdamW
- initial learning rate: 0.0001
- batch size: 16
- GPU: V100

### Semantic Segmentation

![](C:\workspace\논문\swin_transformer\table3.png)

## Ablation study

![](C:\workspace\논문\swin_transformer\table4.png)

## 속도 비교

![](C:\workspace\논문\swin_transformer\table5.png)