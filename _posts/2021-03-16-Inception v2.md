---
layout: post
title: Inception
data: 2021-03-16
excerpt: "Inception"
tags: [Inception, network]
coments: false
mathjax: true
---

# Inception v2, v3

Inception-v2, v3는 GoogLeNet의 후속연구로 2016 CVPR에 *Rethinking the Inception Architecture for Computer Vision*이라는 제목으로 발표가 되었으며, 21년 3월 기준 약 1.2만회가 인용되었습니다.

저자는 GoogLeNet의 저자인 Christian Szegedy 외 4인입니다.

논문에서 Inception-v2와 v3를 동시에 소개하였습니다.

## Introduction

‘VGGNet 은 간단하다고 하지만, 무겁다는 사실은 어쩔수 없다고 합니다. 실제로 GoogLeNet은 AlexNet보다 parameter가 12배 작지만, VGG는 AlexNet보다 parameter가 3배 더 많기 때문입니다.

GoogLeNet은 합리적으로 cost를 사용하였기 때문에 미래를 보면 자신들이 연구한 방향이 맞지만, 이것을 활용하기에는 너무 복잡하다고 합니다. 

GoogLeNet을 확장시키기 위해 단순히 filter의 size를 늘리는 것은 cost가 빠르게 증가하기 때문에 이것은 자신들의 연구방향과 맞지 않으며 비합리적이라고 합니다.

따라서 논문에는 network를 확장할 때, 최적화 하는 방법에 대해 방향성을 제시하였습니다.

## General Design Principles

최적화 하는 방법으로 4가지 원칙을 제시하였습니다. 소개된 4가지의 원칙은 이론적인 부분이 있기 때문에 추후에 검증이 필요하다고 하였습니다.

1. Avoid representational bottlenecks

representational bottleneck이란 pooling으로 인해 feature map의 size가 줄어들어 information의 양이 줄어드는 것을 의미합니다.

즉, 이미지에 무식하게 pooling을 많이 하여 사라지는 information을 우려하고 있습니다.

> 이론적으로 information은 correlation structure와 같은 정보를 버리게 되고, dimensionality는 이런 내용의 추정치를 제공한다.

2. Higher dimensional representations are easier to process locally within a network.

고차원적으로 표현된 정보는 더 많은 요소들로 인수분해 되어 데이터를 표현할 수 있다고 합니다.

이렇게 되면 학습이 더 빨라지며, 메모리의 효율도 향상된다고 합니다.

3. Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power.

Spatial aggregation은 이미지를 축소하는 것을 의미합니다.

즉, 모델의 초반에 lower dimensional embeddings에서는 이미지의 사이즈를 크게 줄여도 괜찮다는 것을 의미합니다.

어느 정도의 축소가 일어나는 것은 인접한 픽셀 정보를 잘 갖고있어주기 때문에 feature map의 사이즈가 작아지는 구간에 진입하기 전까지 사이즈를 줄여주는 것이 빠른 학습에 도움을 줄 수 있다고 합니다.

4. Balance the width and depth of the network

모델의 depth와 width의 밸런스를 잘 맞추어서 함께 최적화가 되어야 좋은 성능을 낼 수 있다고 하였습니다.



저자들은 이 규칙이 성능을 확실하게 올린다는 보장은 없기때문에 적절히 상황에 맞게 사용하면 된다고 하고있습니다.

## Factorizing Convolutions with Large Filter Size

적절한 인수 분해를 사용하여, 더 빠른 훈련을 할 수 있으며, 계산 cost와 메모리가 절약되어 parameter의 수가 감소하게 된다고 합니다.

### Factorization into smaller convolutions

Inception block에서 커다란 conv filter(5\*5 or 7\*7)를 더 작은 conv filter로 대체하였으며, 이러한 방식을 Factorize라고 하였습니다.

5\*5 conv filter를 3\*3 conv filter 2개로, 7\*7 conv filter를 3\*3 conv filter 3개로 Factorization 하였습니다.

> 5\*5 를 3\*3 2개로 바꾸었을때 약 18%의 reduction효과가 있다.

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/inception%20v2/inception-v2_fig3,4.png?raw=true)

위와 같이 conv filter를 대체하였을 경우 표현력 손실 및 linear activation을 사용하는 것이 제안되었습니다. 저자들은 다양한 실험을 통해 rectified linear units를 사용하는 것이 항상 효과가 좋았다고 합니다.

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/inception%20v2/inception-v2_fig2.png?raw=true)

### Spatial Factorization into Asymmetric Convolutions

한번더 Factorization을 진행합니다.

n\*n conv는 1\*n, n\*1 로 이루어진 총 두개의 conv로 대체될 수 있습니다.

따라서, 3\*3 filter를 Asymmetric한 filter 3\*1 과 1\*3 두개로 한번 더 분해합니다.

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/inception%20v2/inception-v2_fig3.png?raw=true)

논문에서는 특히 feature map의 size가 12~20인 구간에서 이 방식의 효과가 좋았다고 하고 있습니다. 따라서 실제 사이즈가 17인 구간에서 inception v2는 아래와 같은 방식을 사용하였습니다.

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/inception%20v2/inception-v2_fig6,7.png?raw=true)

## Utility of Auxiliary Classifiers

저자들은 실험을 통해 모델의 초-중반부의 backprop을 좀 더 촉진 시켜주기 위해 달았던 auxiliary classifiers가 효과가 없는것을 발견하였습니다.

또한 lower auxiliary branch를 제거하여도 결과에 나쁜 영향을 주지 않았습니다.

따라서 저자들은 이 auxiliary classifiers가 성능향상에 도움되는 것보다 Regularizer로 사용되는 것이라고 하였습니다.

## Efficient Grid Size Reduction

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/inception%20v2/inception-v2_fig9.png?raw=true)

왼쪽의 방법은 먼저 pooling을 통해 이미지를 줄이고 inception을 통과하는 방법이고, 오른쪽은 inception을 통과 후 pooling을 통해 이미지를 줄이는 방법입니다.

하지만 왼쪽의 경우는 General Design Principles의 첫 번째를 위반하고 있으며, 오른쪽의 방법은 계산량이 왼쪽에 비해 많아지게 됩니다.

따라서 저자들은 연산량도 줄이고, Representational bottleneck을 피하기 위한 아래와 같은 방법을 제안하였습니다.

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/inception%20v2/inception-v2_fig10.png?raw=true)

inception block안에서 stride 2를 사용하여 이미지 사이즈를 줄이면서 연산량을 줄이는 방법을 사용하였습니다.

GoogLeNet에서는 inception block의 input과 output의 사이즈가 같았지만, 본 논문에서는 다르다는 차이점이 생기게 되었습니다.



## Inception-v2

위의 아이디어가 적용되어 최종적으로는 다음과 같은 구조의 inception-v2가 탄생하게 되었습니다.

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/inception%20v2/inception-v2_table1.png?raw=true)

처음부터 conv layer가 3개가 있는데 이것은 원래 GoogLeNet에서 7\*7 filter를 Factorization을 적용했기 때문입니다.

feature map size가 35\*35부터 inception block을 사용하며 다음단계로 넘겨주게 됩니다.

최종적으로 총 layer는 42개가 되었고, GoogLeNet보다 parameter가 2.5배 많아졌습니다.

## Model Regularization via Label Smoothing

논문에서 Regularization 방법 중 한가지인 label smoothing에 대해서 소개합니다.

train도중에 label-dropout의 효과를 추정하기 위한 것이다라고 말하고 있습니다.

> 참조
>
> https://3months.tistory.com/465
>
> https://data-newbie.tistory.com/370
>
> https://blog.si-analytics.ai/21

다시말해서 label smoothing은 overconfidence를 해결하기 위한 방법입니다. 

즉, 모델이 GT를 정확하게 예측하지 않아도 되고, 정확하지 않은 학습 데이터셋에 치중되는 경향을 막아주어 calibration 및 regularization 효과를 갖게 할 수 있는 방법입니다.

보통 사용하는 방법은 hard label로 정답은1, 나머지는 0으로 구성하는 방법인데, 이것을 soft label로 바꾸는 것입니다.

soft label은 0~1사이의 값을 갖는것을 의미합니다.

예를들어 hard_label의 경우 [0 ,1 0, 0, 0] 과 같이 나왔다면, soft_label은 [0.02, 0.92, 0.02, 0.02, 0.02] 처럼 바꿔주는 것입니다.

s이렇게 해주면, 데이터셋이 mislabeling된 경우를 잘 넘어갈 수 있다고 합니다.

그래서 이렇게 스무딩된 소프트 라벨을 cross-entropy를 통해 기존 라벨을 대체해 사용하게 됩니다.

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/inception%20v2/inception-v2_cross.png?raw=true)

위의 식과 같이 $y_k$대신 $y^{LS}_k$를 사용해서 cross-entropy loss를 최소화 하여 라벨 스무딩 효과를 적용할 수 있습니다.



본 논문에서는 label smoothing regularization을 cross-entropy에 적용한 식을 다음과 같이 사용하였습니다.

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/inception%20v2/inception-v2_cross_soft_label.png?raw=true)

- K: 전체 label 수
- u: label의 분포
- $\epsilon$: smoothing parameter, 0.1 사용

이를 통해 ImageNet2012의 validation에 대해 0.2%의 성능향상이 있었다고 합니다.



## Training Methodology

 NVidia Kepler GPU 50개를 사용하여 학습 시켰으며 파라미터는 다음과 같습니다.

- Batch size : 32
- Epochs : 100
- Optimizer : RMSProp with decay of 0.9
- LSR : $\epsilon$=1.0
- Learning rate : 0.045 decayed every 2 epoch using an exponential rate of 0.94

## Performance on Lower Resolution Input

일반적으로 해상도가 높을수록 성능이 높게 나타나는 경향이 있습니다. 

논문에서는 작은 해상도에서도 성능이 좋게 나오는지 확인하기 위해 이미지의 해상도를 변경하며  Inception-v2 구조를 통해 실험을 진행하였습니다.

사용한 해상도는 79\*79, 151\*151, 299\*299 총 3개의 해상도를 사용하였고, 각기 다른 방법으로 실험을 진행하였습니다.

299\*299의 해상도인 경우 첫번째 레이어를 stride 2로 통과 한 후, maxpooling을 해주었습니다.

151\*151의 해상도인 경우 첫번째 레이어를 stride 1로 통과 한 후, maxpooling을 해주었습니다.

79\*79의 해상도인 경우 첫번째 레이어를 stride 1로 통과 후, maxpooling은 하지 않았습니다.

그 결과 아래와 같은 Top-1 Accuracy가 나타났습니다.

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/inception%20v2/inception-v2_table2.png?raw=true)

결과를 보면 세개의 해상도 모두 비슷한 정확도를 나타내는 것을 확인할 수 있었습니다.

즉, 모델의 구조를 이미지 해상도에 맞게 잘 조절한다면 커다란 성능저하가 없다는 것을 보여줍니다.

## Experimental Results and Comparisions

위에서 설명한 기법들은 하나하나 섞어가며 실험을 진행하였습니다.

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/inception%20v2/inception-v2_table3.png?raw=true)

table3을 보면 많은 기법들을 섞어서 비교하였는데요

제일 마지막줄은 Inception-v2 + BN-auxiliary + RMSProp + Label Smoothing + Factorized 7\*7을 한 것입니다.

이것의 경우 Top-1 Error가 21.2%, Top-5 Error는 5.6%로 표에 있는 모델중 error율이 가장 낮은것을 확인할 수 있고, 저자들은 이것은 Inception-v3라고 명명하였습니다.



저자들이 General Design Principles를 통해 세웠던 가설들을 하나씩 적용했을때, 실제로 성능이 좋아진 것을 확인할 수 있었습니다.

