---
layout: post
title: Deformable DETR
data: 2021-07-10
excerpt: "Deformable DETR"
tags: [transformer]
coments: false
mathjax: true
---

# Deformable DETR

- 제목: Deformable DETR: Deformable Transformers for End-to-End Object Detection
- 저자: Xizhou Zhu 외 5명
- 인용수: 129회
- 학술지: ICLR 2021

## Introduction

- DETR은 새로운 object detector로써 좋은 성능을 나타내었지만 다음과 같은 문제점이 있음

  1) 현재 존재하는 detector보다 더 많은 epoch을 요구(500epoch)

  - 어떠한 query에 대한 key의 weight가 $1/N_k$이기 때문에 $N_k$가 클 경우 ambiguous gradient가 발생

  - 학습 시작시 attention 모듈은 feature map pixel에 대해 동등한 weight를 부여함

  2) samll object에 대해 성능이 떨어짐

  - 이 문제에 대해 현재 detector들은 multi-scale feature map을 사용하거나 높은 해상도를 사용

  - DETR에 계산 복잡도는 feature map 길이의 제곱에 비례하기 때문에 높은 해상도를 사용하기 어려움

  

- 본 논문에서는 이 문제들을 Deformable Convolution와 attention을 결합한 Deformable attention moduel을 제안하여 해결하고자 함

![](.\deformabledetr_fig1.png)

## Deformable Convolution

- 일반적인 convolution을 3\*3, 5\*5 등 고정된 필터를 사용하기 때문에 다양한 형태에 사용하는데 한계가 있음
- convolution의 필터모양을 다양한 형태로 변형시키도록 offset을 추가로 학습 시킨 후 이를 갖고 convolution을 진행
- object scale에 따라 receptive field가 달라지게됨

![](.\deformableconv_fig2.png)

![](.\deformableconv_fig5.png)

## Deformable Attention Module

![](.\deformabledetr_fig2.png)

- 기존의 attention module은 입력된 모든 feature map에 대해 수행함

![](.\deformabledetr_s1.png)

> $z_q$=input feature of $q^{th}$ query
>
> $x$: input feature map
>
> M: number of attention heads
>
> $A_{mqk}$: attention weight of $q^{th}$ query to $k^{th}$ key at $m^{th}$ head
>
> $x_k$: input feature of $k^{th}$ key

-  deformable attention module은 deformable convolution을 도입한 attention으로 feature map에서 k개 만큼 sampling을 하여 attention을 수행함

![](.\deformabledetr_s2.png)

> $z_q$=input feature of $q^{th}$ query
>
> $p_q$: reference point for $q^{th}$ query
>
> $\triangle{p_{mqk}}$: sampling offset of $q^{th}$ query to $k^{th}$ key at $m^{th}$ head
>
> $x$: input feature map
>
> M: number of attention heads
>
> $A_{mqk}$: attention weight of $q^{th}$ query to $k^{th}$ key at $m^{th}$ head

- 이를 multi scale에 대해 적용하기 위해 level을 추가한 형태의 attention module을 사용

![](.\deformabledetr_s3.png)

![](.\deformableconv_fig4.png)

- resnet의 C3~C5를 input으로 사용

## Encoder

- attention module를 multi-scale deformable attention module로 바꾸어 사용
- C3~C5의 feature map을 사용

## Decoder

- cross attention과 self attention의 두 가지 module이 존재
- 본 논문에서는 cross attention을 multi-scale attention으로 바꾸어 사용하였음

- reference point에 대해 offset을 학습하게 됨
- 따라서 마지막 FFN에서는 bbox의 좌표가 나오는 것이 아닌 box에 대한 offset을 출력하게 됨(?)

## Experiment

- Use dataset: COCO 2017
- Use backbone: ImageNet pre-trained ResNet-50
- M = 8
- K = 4
- N = 100~300

![](.\deformabledetr_table1.png)

![](.\deformabledetr_fig3.png)

- Deformable DETR은 기존의 DETR보다 훨씬 더 적은 epoch을 사용하여 동일한 성능을 내었음
- small object에 대해서 더 높은 성능을 나타냄
- 더 적은 parameter인데도 불구하고 FPS가 낮게 되었는데, 이 이유로 저자들은 Deformable DETR이 memory에 랜덤하게 액세스가 되기 때문에 최적화 문제라고 언급하였음

![](.\deformabledetr_table2.png)

- FPN을 사용하여도 효과를 보기는 어려웠음

  > 이미 multi-scale attention이 FPN의 역할을 하고 있기 때문