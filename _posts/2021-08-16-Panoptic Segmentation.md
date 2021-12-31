---
layout: post
title: Panoptic Segmentation
data: 2021-08-16
excerpt: "Panoptic Segmentation"
tags: [segmentation]
coments: false
mathjax: true
---

# Panoptic Segmentation

- 제목: Panoptic Segmentation
- 저자: Alexander Kirillov, Kaiming He, Ross Grishick 외 2명
- 인용수: 432
- 학술지: CVPR 2019

## Introduction

- Sementic Segmentation은 이미지의 모든 pixel을 분류하지만, 해당 pixel의 class가 중요하며, 해당 class의 양이 어느정도 되는지는 중요하지 않음

- Instance Segmentation은 관심있는 부분(object)의 pixel을 분류하며 해당 class의 양이 어느정도인지 확인 가능하지만, 배경과 같은 셀 수 없는 부분에 대해서는 분류를 진행하지 않음

- 본 논문에서는 Sementic Segmentation과 Instance Segmentation을 합친 Panoptic Segmentation을 제시

- Panoptic Segmentation은 Stuff와 Things을 구분할 수 있도록 함

  > Stuff: Semantic Segmentation으로 다룰 수 있으며, 형태가 없는 셀 수 없는 영역
  >
  > Things: Instance Segmentation으로 다룰 수 있으며, 셀 수 있는 영역

![](C:\workspace\논문\panoptic segmentation\fig1.png)

- 기존의 metric은 stuff와 things를 동시에 고려하지 않았기 때문에 이를 평가할 Panoptic Quality(PQ)라는 새로운 metric을 제시함



## Panoptic Segmentation Format

- 이미지의 각 pixel을 $(l_i, z_i)$쌍으로 mapping진행하는데, 이때, $z_i$는 같은 class의 pixel group을 구분된 segment로 다시 나눔

  > $l_i$: i번째 pixel의 Sementic class
  >
  > $z_i$: i번째 pixel의 Instance id

- Label $L$은 $L^{st},\space L^{th}$ 로 나뉘게 되며 두 label의 교집합 부분은 없음. 즉, $L^{st}$로 label된 pixel의 경우 $L^{th}$에서 제외됨

  > $L=L^{st}+L^{th}$

- $L^{st}$의 경우 모든 pixel을 동일한 instance로 취급함

  > e.g. 하늘, 나무, 도로

- $L^{th}$의 경우 하나의 instance에 속하는 pixel은 모두 동일한 class로 할당되어야 함

  > 하나의 object에 있는 pixel은 여러개의 sementic class로 할당. 즉, 하나의 instance id에 대해 sementic class는 하나로 할당 될 수 있음
  >
  > e.g. 차, 사람, 표지판

- stuff의 경우 이미지의 각 pixel에 sementic label이 할당되어 있어야 하며, instance의 경우 각 pixel마다 하나의 sementic label과 instance label이 $(l_i, z_i)$쌍으로 할당되어야 함

  > stuff의 경우 instance label이 할당되어 있지 않아도 sementic label은 할당되어 있어야 하며, things의 경우 sementic label과 instance label이 존재하여야 함

- Panoptic Segmentation의 경우 Confidence Score가 필요하지 않음. 단, score가 더 많은 정보를 제공할 가능성도 있기 때문에 필요시 사용을 하여도 됨



## Panoptic Segmentation Metric

- 간단하면서 구현이 쉬운 새로운 panoptic quality metric을 소개함
- 이 metric엔 두 스텝이 포함되어 있음
  1. segment matching
  2. PQ 계산

### Segment Matching

- 이미지를 통해 예측된 segment가 주어지면 그에 해당하는 gt에 대해 최대 하나의 segment가 할당 (설정한 IoU값을 기준으로 e.g. IoU > 0.5)

### PQ Computation

- PQ metric은 클래스 마다 계산한 다음 클래스에 대한 평균을 계산함

  > class imbalance의 경우에도 유의미한 값을 가질 수 있게 됨

- 각 클래스 별로 predict와 gt의 segment는 세 가지의 set을 갖고 있음

  > TP, FP, FN

  ![](C:\workspace\논문\panoptic segmentation\fig2.png)

  

- PQ는 다음과 같이 계산됨

![](C:\workspace\논문\panoptic segmentation\s1.png)

- 분모의 FP와 FN을 제외하고 보면 다음과 같이 볼 수 있음

![](C:\workspace\논문\panoptic segmentation\s2.png)

- 이 부분은 단순히 TP로 매칭된 segment의 평균 IoU를 나타냄
- 나머지 FP와 FN이 포함된 부분은 매칭이 되지않은 segment에 대해 패널티를 주는 역할을 함
- 또한 PQ는 segmentation quality(SQ) + recognition quality(RQ)로 표현이 될 수 있음

![](C:\workspace\논문\panoptic segmentation\s3.png)

- RQ의 경우 F1 score의 형태임

  > F1 score: precision과 recall의 조화평균
  > $$
  > 2*\frac{Precision*Recall}{Precision+Recall}
  > $$
  > precision과 recall의 차이가 클 때 모델이 얼마나 효과가 있는지에 대해 단순평균으로는 올바르지 않기 때문에
  >
  > e.g. precision:1, recall:0.01
  > $$
  > 단순평균:\frac{1+0.01}{2}=0.505\\
  > 조화평균:\frac{1*0.01}{1+0.01}*2=0.019
  > $$

- void label에 대해서는 성능을 측정하지 않았음
  - 매칭 동안에서는 gt에서 void label인 경우 예측에서 제거
  - 매칭이 완료된 후에서는 pred에서 threshold는 만족하지만, void pixel이 포함된 경우 포함된 segment를 제거

### Comparision to Existing Metrics

- Semantic segmentation metric의 경우 각 클래스에 대해 pixel level에서만 성능을 계산(object level에 대해서는 무시)하는데, 이때 IoU는 옳게 예측한 pixel수와 총 pixel수의 비율로 성능을 계산함. 즉, pixel의 정확도를 나타냄

  > Prediction, GT에 대해서

- Instance segmentation metric의 경우 Average Precision(AP)를 사용하는데, 이를 Sementic Segmentation에 적용하기에는 어려움

## Result

![](C:\workspace\논문\panoptic segmentation\fig9.png)



![](C:\workspace\논문\panoptic segmentation\table3,4.png)

![](C:\workspace\논문\panoptic segmentation\table5.png)

![](C:\workspace\논문\panoptic segmentation\table6.png)