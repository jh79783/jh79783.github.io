---
layout: post
title: EfficientDet
data: 2021-05-19
excerpt: "EfficientDet"
tags: [EfficientDet, detector]
coments: false
mathjax: true
---

# EfficientDet

- 제목: EfficientDet: Scalable and Efficient Object Detection
- 저자: Mingxing Tan 외2명 (Google Research, Brain Team)

- 인용수: 491회
- 학술지: CVPR 2020

## Introduction

- Speed와 Accuracy는 Trade-Off관계이기 때문에 두 가지 효율을 모두 잡는 것은 어려움
- Object Detection에서 두 가지의 효율을 모두 잡는 모델을 설계하는 것이 목표
- 이를 위하여 2개의 Challenge를 정함

## Challenge 1: efficient multi-scale feature fusion

- FPN은 현재 대부분의 Object Detection에서 사용되고 있음
- 성능을 개선하고자 하는 다양한 연구(PANet, NAS-FPN 등)에서 성능을 개선하고자 하는 연구가 많이 진행
- 하지만 이런 연구들은 input feature의 크기가 다른 경우에서 합칠 때 단순히 더하는 방식에 대해 문제점을 제기하였음
- 본 논문에서 feature의 크기에 따라 output feature에 기여하는 정도를 다르게  하는것을 주장
- 주장에 따라 weighted bi-directional FPN(BiFPN) 을 제안
- BiFPN을 사용하여 크기가 다른 input feature에 따라 output feature에 기여하는 정도를 학습이 가능하였음

## Challenge 2. Model scaling

- 기존에는 매우 큰 backbone network를 사용하거나 큰 input image를 사용하여 높은 정확도의 성능을 달성함
- 본 논문에서는 compound scaling 방법을 사용하여 높은 성능을 달성함
  - Compound Scaling: 모델의 크기와 연산량을 결정하는 요소(input resolution, depth, width)를 동시에 고려하는 방법
- 이를 모든 backbone, feature network, box/class prediction network에 적용
- backbone은 EfficientNet을 사용하고, BiFPN과 compound scaling을 결합하였음

# BiFPN

## Cross-Scale Connections

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/efficientdet/EfficientDet_Fig2.png?raw=true)

- (a): 전통적인 FPN의 구조
  - 하향식 구조로 정보가 단방향으로 흐르기 때문에 제한적임
- (b): FPN의 구조를 개선한 PANet
- (c): 가장최근의 NAS-FPN
  - AutoML의 Neural Architecture Search를 위해 FPN 구조에 적용한 것
  - GPU를 사용하면서 수천시간이 필요
  - 검색이 완료된 network는 불규칙하고 해석하기 어려움
- (d): 본 논문에서 제안한 간단하면서 효율적인 BiFPN
  - 입력인 하나인 경우 기여도가 적다고 판단하여 삭제 함
  - 보라색 선과 같이 같은 scale에서 입력을 추가하여 적은 cost로 더 많은 feature가 fusion되도록 함
  - 이를 여러번 반복하여 사용

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/efficientdet/EfficientDet_table4.png?raw=true)

## Weighted Feature Fusion

- 다른 크기의 input feature을 합치기 위해 resize를 통해 같은 크기로 만들어준 뒤 합치는 방법을 사용 중
- 이것이 문제임을 발견하고, 이를 개선하기 위해 각각의 input feature에 가중치를 주어 학습을 통해 가중치를 학습할 수 있는 3가지의 방법 제안

### Unbounded fusion

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/efficientdet/EfficientDet_unbounded.png?raw=true)

> I: feature map
>
> w: scalar weight

- 단순히 weight를 곱해주어 sum하는 방식
- weight를 scalar로 사용하는 것이 실험을 통해 정확도와 연산 측면에서 효율적임을 확인하여 scalar weight를 사용
- undound 되어있기 때문에 학습에 불안정성이 발생할 수 있음

### Softmax-based fusion

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/efficientdet/EfficientDet_softmax.png?raw=true)

- 0~1사이의 범위로 변환하여 sum하는 방식
- GPU에서 연산시 속도가 많이 떨어지는것을 확인

### Fast normalizaed fusion

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/efficientdet/EfficientDet_fast.png?raw=true)

- 본 논문에서 채택한 fusion method
- $w_i$가 양수이여야 하기 때문에 Relu를 통과한 값을 사용
- 분모가 0으로 나뉘는 것을 방지하기 위해 $\epsilon$( 0.0001)을 추가함
- 실험 결과 softmax방식과 성능의 차이는 크게나지 않지만, 속도측면에서 30%향상된 모습을 나타냄

# EfficientDet

### EfficientDet Architecture

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/efficientdet/EfficientDet_Fig3.png?raw=true)

- Backbone: EfficientNet-B0 ~ B6

  > B~: AutoML을 사용하여 찾은 모델

- Feature Network: BiFPN

- class와 box의 weight는 서로 공유함

### Compound Scaling

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/efficientdet/EfficientDet_compound.png?raw=true)

- width scaling
  - filter의 수를 늘려 scale-up하는 방법
  - 대게 작은 크기의 모델들이 width를 제어
  - width를 넓게 할수록 fine-grained feature를 더 많이 담을 수 있는 것이 확인됨
- depth scaling
  - layer의 수를 늘려 scale-up하는 방법
  - 깊을수록 더 좋은 성능을 나타내나 한계가 있음
- resolution scaling
  - input image의 해상도를 높여 scale-up하는 방법
  - object detection영역에서 600\*600을 사용할 때 좋은 성능을 나타내는 것을 확인
- compound scaling
  - EfficientNet에서 제안하는 방법
  - width + depth + resolution을 적절히 조절해 정확도를 높이는 방법

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/efficientdet/EfficientDet_compound_scale.png?raw=true)

- 모든 scaling에 대한 값을 추정하기 어려워 heuristic-based scaling approach사용

- Backbon의 compound scaling

  - width, depth는 efficientnet b0~b6과 같이 설정

  -  input은 다음과 같음

  - $R_{input}=512+\phi*128$으로 설정

    > resolution은 level3~7에서 사용되어 $2^7$을 사용

- BiFPN의 compound scaling

  - $W_{bifpn}=64*(1.35^\phi)$로 설정

    > 실험을 통해 1.35가 heuristic하게 정수가 나오도록 설정

  - $D_{bifpn}=2+\phi$

- Box/Class prediction의 compound scaling

  - $W_{pred}=W_{bifpn}$
  - $D_{box}=D_{class}=3+[\phi/3]$

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/efficientdet/EfficientDet_table1.png?raw=true)

> $\phi$에 따른 각각의 정보

# Experiments

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/efficientdet/EfficientDet_fig1.png?raw=true)

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/efficientdet/EfficientDet_fig4.png?raw=true)