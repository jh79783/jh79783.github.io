---
layout: post
title: RetinaNet
data: 2021-05-01
excerpt: "RetinaNet"
tags: [retinanet, network]
coments: false
mathjax: true
---

# RetinaNet

- 제목: Focal Loss for Dense Object Detection
- 저자: Tsung-Yi Lin 외 4명 (Facebook AI Research)

- 학술지: ICCV2017

## Class Imbalance

- one-stage detector의 학습과정에서 class imbalance가 가장 큰 문제임을 확인
  - 한장의 이미지에서 $10^4$~$10^5$개의 후보 위치를 제안 하지만, 실제로 object는 수개 정도만 존재
- class imbalance가 일으키는 문제
  - 실제 후보 위치는 분류하기가 쉬운 easy negative가 대부분이기 때문에 train에서 비효율적
  - easy negative가 굉장히 많아 train과정을 이끌다 보니 model의 성능이 낮아지는 경우가 발생
- 이를 해결하기 위해 일반적으로 **hard negative mining**을 사용
- 본 논문에서는 제안하는 focal loss를 사용하면 class imbalance문제를 자연스럽게 해결 가능

## Robust Estimation

- hard example에서의 loss영향을 줄이기 위한 robust loss function
- 논문에서 제안하는 focal loss는 robust loss function과는 정 반대인 easy example에서의 loss영향을 줄이고 hard example에 대해 중점을 둠

## Focal loss

- Focal loss는 one-stage detector에서 지나친 class imbalance문제를 해결하기 위해 디자인 됨

- 일반적인 cross entropy의 공식은 다음과 같음

![](.\retinanet_s1.png)

![](.\retinanet_s2.png)

## Balanced Cross Entropy

- cross entropy의 imbalance문제를 해결하기 위해 $\alpha$를 곱함
  - $\alpha$는 0~1사이의 값을 갖고있는 가중치 파라미터

![](.\retinanet_s3.png)

- positive/negative sample사이의 균형은 잡아주지만 easy/hard sample에 대해서는 잡아주지 못함

## Focal Loss

- traing과정에서 class imbalance문제가 심각한 경우, 예측하기 쉬운 negative sample이 많아 gradient에 큰 영향을 주게 됨
- Focal Loss는 easy example의 중요도를 낮추어 hard negative에 집중하게 만들어 줌
- Focal Loss는 cross entropy에 modulating factor인 $(1-p_t)^\gamma$와 tunable focusing paramter인 $\gamma$를 추가한 형태

![](.\retinanet_s4.png)

- $\gamma$값의 에 따른 loss는 다음과 같음

![](.\retinanet_fig1.png)

- pt와 modulating foctor의 관계

  - example이 잘못 분류되고 pt가 작은 경우 loss에 영향을 받지 않게되며, example이 잘 분류되고 pt가 큰 경우 loss는 영향력이 적게 됨

- focusing parameter의 역할

  - 부드럽게 easy example의 영향력을 줄여줌

  - 즉, easy example의 loss의 영향력을 줄이고, loss를 낮추게 만드는 example의 범위를 결정해주며, 잘못 분류된 example의 중요도를 높이는 역할을 해줌

    > ex) $\gamma=2, p_t=0.9$
    >
    > 기존 cross entropy에 비해 100배 더 적은 loss를 갖게 됨

  - $\gamma=0$인 경우 cross entropy와 같으며 커질수록 modulating factor의 영향이 크게됨

  - 본 논문에서는 $\gamma=2$일때 가장 좋은 결과를 나타내었음

## Class Imbalance and Model Initialization

- 기본적인 binary cliassification model은 class y가 -1, 1일 확률이 동일하도록 초기화 (0.5, 0.5)

- 이때 class imbalance가 나타면 학습 초기부터 불안정해지는 현상 발생

  > loss가 nan으로 나오는 현상

- 이를 해결하기 위해 'prior'라는 개념의 p항을 사용

- p항은 rare class(foreground)의 비율로 최초 학습시에 사용하여 학습 안정화에 도움

  - 어떤 class에 대해 binary cross entropy를 사용하기 위해 softmax함수를 이용

    > 어떤 class가 맞을, 틀릴 확률을 나타내는데 이 두 확률을 0.5가 아닌 0.01로 초기화 하고 싶다!

  - softmax 이후의 값을 0.01로 고정 시키면 해결

  ![](.\retinanet_initialization.png)

  - bias는 다음과 같이 초기화

  ![](.\retinanet_bias_init.png)

  - 즉, bias를 위의 공식으로 초기화하고, weight는 (0, 0.01)에서 sampling하여 초기화

    > convolution weight는 0에 가깝고 bias는 b가 되어 bias만 남게됨
    >
    > 따라서 softmax 이후의 값을 0.01로 제한이 가능

## RetinaNet Detector

- backbone network와 2개의 subnetwork로 구성

![](.\retinanet_fig3.png)

## Feature Pyramid Network Backbon

- ResNet + FPN을 backbone으로 사용

- 피라미드의 레벨을 P3~P7까지 5개의 scale을 사용

  > ex)$P3 = input size/2^{3}$

- 모든 피라미드의 C(채널)는 256

> input: image
>
> output: feature pyramid(P3~P7)

## Anchor

- anchor box의 크기는 피라미드 레벨 P3~P7에서 $32^2$~$512^2$ 사용
- 각 scale마다 3개의 비율을 갖는 box 사용
  - aspect ratio: 1:2, 1:1, 2:1
  - denser scale coverage 하기 위해 원래의 ratio box에 각각 3개의 size box 추가
  - size: $2^0, 2^{1/3},2^{2/3}$
  - scale당 9개의 box를 갖고있어 32~813 픽셀 담당
- GT와 IOU가 0.5이상인 box만 GT로 할당
- IOU가 0.4이하인 박스는 background로 할당
- 0.4~0.5의 IOU는 무시



## Classification Subnet

- FPN에 작은 FCN을 붙인 것
- fig3의 C와 같이 backbone의 feature map을 input으로 받음
- 4개의 3\*3 conv layer가 있으며, C개의 filter를 적용
- activation function은 ReLU사용
- 마지막엔 3\*3conv를 K\*A의 filter로 convolution 진행
  - K: class 갯수
  - A: 총 anchor box의 갯수 - 9개

> input: backbone으로부터의 feature map
>
> output: 5개 scale의 feature map (K\*A channel)

## Box Regression Subnet

- GT와 anchor box의 offset예측(cx, cy, w, h)
- classification의 subnet과 공통의 구조를 갖지만, parameter는 공유하지 않음

> input: backbone으로부터의 feature map
>
> output: 5개 scale의 feature map (4\*A channel)

## Inference

- RetinaNet의 속도 향상을 위해 FPN에서 점수가 가장 높은 1000개의 box만 사용
- NMS threshold = 0.5

## Focal Loss

- focal loss는 각 이미지의 10만개의 anchor box에 적용되며, 모든 박스의 loss합으로 계산
- positive와 negative sample을 구분하기 위해 $\alpha$를 적용
- $\gamma$를 키울수록 $\alpha$를 감소시켜야함
- $\gamma=2, \alpha=0.25$일때가 가장 성능이 좋았음

![](.\retinanet_figb.png)

## Experiments

- COCO dataset으로 평가

![](.\retinanet_fig2.png)

- 다른 sota detector와 비교한 결과

![](.\retinanet_table2.png)