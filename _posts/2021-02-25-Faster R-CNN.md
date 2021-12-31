---
layout: post
title: Faster R-CNN
data: 2021-02-25
excerpt: "Faster R-CNN"
tags: [r-cnn, network]
coments: false
mathjax: true
---

# Faster R-CNN

- 제목: Towards Real-Time Object Detection with Region proposal Networks
- 저자: Shaoqing Ren 외 3명
- 인용수: 약 2만회

## Abstract

SPPnet과 Fast R-CNN은 bottleneck을 유발하는 region proposal(RP)계산을 사용하였습니다.

본 논문에서는 기존의 RP를 대신할 Region Proposal Network(RPN)을 소개합니다.

RPN은 Fully Convolutional Network로 bbox와 classifier를 동시에 예측이 가능합니다.

Faster R-CNN은 Fast R-CNN의 구조를 그대로 계승하였기 때문에 결과적으로 본 논문은 RPN과 Fast R-CNN의 feature map을 공유하는 한 개의 네트워크로 합친 것입니다.

VGG-16 모델을 GPU를 통해 실험하였을때, 5fps를 가졌으며, PASCAL VOC2007, 2012, COCO 데이터를 사용하여 SOTA를 달성하였습니다.

## Introduction

기존에 사용하던 RP를 찾는 방법은 Selective Search를 사용하였습니다. 하지만 이 방법은 이미지당 2초씩 걸리는 느린 속도를 나타내었습니다.

이를 대신할 EdgeBoxes라는 방법이 등장하여 이미지당 0.2초로 감소되었지만 여전히 detection부분에서는 오래걸렸습니다.

이 문제를 RPN을 통하여 알고리즘적으로 해결하였습니다.

테스트시에 convolution들을 공유하여 계산시간을 감소시켜, 결과적으로 이미지당 10ms가 나왔습니다.

저자들은 Fast R-CNN에서 feature map이 RP로 사용될 뿐만 아니라 RP를 생성하는 것을 확인하였기 때문에 feature map이 만들어졌을 때, RPN을 feature map으로 쌓고 정형화된 grid로 나누면 bbox와 classifier를 동시에 할 수 있을 것이라고 보았습니다.

![](.\faster rcnn_fig1.png)

일반적으로 (a)나 (b)와 같은 방법으로 aspect ratio에서 RP를 예측하였습니다.

본 논문에서는 (c)와 같이 Archor라는 다양한 scale과 aspect ratio를 가진 박스를 소개하였습니다.

RPN은 RP와 object detection을 위한 fine-tuning을 선택적으로 학습시킬 수 있으며, 빠르게 수렴하였습니다.

결과적으로 ILSVRC와 COCO2015대회에서 ImageNet detection, ImageNet localization, COCO detection, COCO segmentation에서 1등을 하였습니다.

## Faster R-CNN

Faster R-CNN은 2개의 모듈로 구성되어 있습니다.

1. PR을 만드는 deep fully convolutional network
2. PR을 사용하기 위한 Fast R-CNN detector

![](.\faster rcnn_fig2.png)

### Region Proposal Networks

RPN의 입력은 (모든 크기의)이미지이고 출력은 objectness score를 포함한 사각형 object proposal 의 set입니다. 이런 출력과정을 fully convolutional network로 모델링 하였습니다.

본 논문의 궁극적인 목표는 Fast R-CNN과 연산을 공유하는 것이기 때문에 두 네트워크의 convolutional 층들을 같은 것으로 사용하였습니다. 실험에서는 ZF model과 VGG 16을 사용하였습니다.

- ZF model의 공유가능한 Conv layers : 5
- VGG 16의 공유가능한 Conv layers: 13

RP를 생성하기위해 공유된 마지막 Conv layer를 통해 출력된 feature map을 작은 network로 sliding하였습니다. 이때, 저차원 feature로 mapping되며, 이 feature들은 두개의 FC layer(box regression layer(reg), box classification layer(cls))로 보내졌습니다.

논문에서는  sliding window를 3으로 두고 사용하였습니다.

#### Anchors

각 sliding window에서 동시에 RP를 예측할 수 있는데 최대의 RP 수는 k로 표시됩니다.

box regression layer는 좌표를 갖고있기 때문에 4k, box classification layer는 확률을 추정하기 때문에 2k로 표시할 수 있습니다.

이 k개의 proposal들은 k개의 reference box에 대한 매개변수로 표시되는데 이를 **Anchor**라고 합니다.

Anchor는 sliding window의 중심좌표를 갖고있으며, scale & aspect ratio와 관련이 있습니다.

기본값으로 3개의 scale과 3개의 aspect ratio를 사용하여 각각의 sliding window마다 9개의 anchor를 도출해 냅니다.

> 보통 feature map은 W\*H이기 때문에 이미지당 W\*H\*K(9)의 anchor를 가지게 된다.

### Translation-Invariant Anchros

논문에서는 Anchor를 사용하는데 있어 Translation-Invariant하다고 설명하고 있습니다.

Translation-Invariant 는 물체가 이미지에서 이동한다면, proposal도 이동되어야하며 같은 proposal을 생성해야 하는 것이라고 합니다.

이것을 다시 말하면 물체가 특정 위치에 존재할 때만 탐지되거나, 특정 위치에서는 탐지가 잘 되지 않는 현상을 줄이는 것이라고 합니다.

이를 Multi Box와 비교하였습니다.

Multi Box method는 k-means를 사용해 800개의 anchor를 생성하지만 이것은 Translation-Invariant특성을 갖고있지 않는다고 합니다.

또한 Translation-Invariant는 parameter수를 줄이게 해주어 model의 size를 작게 할 수 있다고 합니다.

> Multi Box의 경우: (4+1)\*800-demensional fully-connected layer를 사용
>
> Faster R-CNN의 경우: (4+2)\*9-demensional convolutional output layer를 사용

그리고 이것은 overfitting문제에 있어서도 Fater R-CNN이 더 우수할 것이라고 보고 있습니다.

### Multi-Scale Anchors as Regression References

![](.\faster rcnn_fig1.png)

사진처럼 multi-scale 예측을 위한 두 가지 방법이 있습니다.

(a)와 같이 이미지 당 feature map이 피라미드 형태로 있는 것인데, 이것은 각 이미지 마다 계산을 해 주어야 하기 때문에 시간이 너무 오래 걸린다는 단점이 있습니다.

따라서 선택된 방법이 (b)와 같은 feature map에서 많은 크기와 비율의 sliding window를 사용하는 방법입니다.

본 논문에서는 Anchor 피라미드를 사용하였습니다. 이 방법을 사용하여 다양한 anchor boxes와 scale, aspect ratio를 이용해 bboxes를 예측하고, 분류하였습니다.

논문에서는 3가지 크기와 3가지 비율을 가진 총9개의 anchor를 사용했습니다.

> (128\*128), (256\*256), (512\*512)
>
> 2:1, 1:1, 1:2

다양한 크기의 anchor box를 사용하여 cost-efficient(추가적인 cost없이)하게 특징을 공유할 수 있게 되었습니다.

#### Loss Function

RPN 학습을 진행할 때, 각 앵커마다 positive or negative의 라벨을 할당해 주었습니다. 각 라벨은 다음과 같습니다.

Positive

- GT box와 가장 높은 IoU를 가지는 경우

- IoU가 0.7보다 높은 경우

Negative

- IoU가 0.3보다 낮은 경우

> 두가지 모두 해당되지 않는 경우 (0.3 < IoU < 0.7) 학습 데이터로 사용하지 않음

이 방법으로 Fast R-CNN의 multi-task loss를 따르게 됩니다.

> ![](.\faster rcnn_s1.png)
>
> i: batch 당 앵커의 index
>
> pi: 앵커 i가 사물로 인식한 확률(score)
>
> pi\*: GT라벨(1:positive, 0:negative)
>
> ti: bbox좌표
>
> ti*: GT에서의 bbox좌표
>
> Lcls: classification loss. 즉, object인지, background인지 나타내는 score
>
> lamda: reg와 cls간 동등한 가중치를 부여하는 기능(기본값 10)
>
> Lreg: smooth L1 방식을 사용해 bbox regression 적용
>
> ![](.\faster rcnn_s2.png)
>
> x, y: 중심좌표
>
> w, h: 너비와 높이
>
> O, O_a, O*: 예측한 box, anchor box, GT box의 값

본 논문에서는 다양한 크기들도 detection하기위해 k 개의 bbox regressor를 학습시켰습니다.

각 regressor는 단일의 크기와 비율을 갖게 되는데 k개의 regressor는 가중치를 공유하지 않습니다. 따라서 feature가 다른 크기나 비율을 갖더라도 detection이 가능하다고 합니다.

#### Training RPNs

Fast R-CNN과 같이 이미지의 중심으로 샘플링을 하게 됩니다. 각 mini-batch로 negative와positive anchor가 포함된 이미지가 생성되는데, 샘플들은 negative로 치우치게 됩니다. 따라서 무작위로 negative와positive샘플을 1:1비율로 256개의 anchor를 뽑아내었습니다.

> 만약 positive 샘플이 128개(50%)보다 적다면 나머지는 negative로 채움

PASCAL VOC 데이터에서 사용한 기본 설정은 다음과 같습니다.

- learning rate(mini-batch 60k): 0.001
- learining rate(mini-batch 20k): 0.0001
- momentum: 0.9

### Implementation Details

이미지를 단일크기로 사용하였으며 600pixel로 재조정하였다고 합니다. 여러 크기를 사용하면 성능은 향상되지만 그만큼 속도가 느려졌습니다.

또한, stride를 사용하면 정확도가 향상되었다고 합니다.

anchor사용하였는데, anchor parameter의 선택은 그렇게 중요하지 않다고 말하고 있습니다.

이미지의 경계에 존재하는 anchor는 무시한다고 합니다. 따라서 loss에 영향을 끼치지 않게됩니다.

> 1000\*600의 이미지는 약 20000개의 anchor를 갖고있어 이미지당 6000개의 anchor가 훈련에 사용됨
>
> 학습을 진행 할 수록 더 많이 쌓이게 되어 나중에는 활용이 불가능해짐

그러나 테스트시 RPN을 전체 이미지에 사용하기 때문에 경계에 proposal box가 생기게 됩니다.

이 box들은 겹쳐있어 이를 없애기 위해 NMS(non-maximun suppression)를 cls에 적용시켰습니다. 이때, NMS의 IoU threshold를 0.7로 고쳐 사용했기 때문에 실제 테스트시에는 몇 개의 proposal만 사용되었습니다.

### Experiments on PASCAL VOC

RPN + Fast R-CNN을 사용하여 300개의 proposal이 생성되었고, 이를 사용해 59.% mAP를 달성하였습니다.

또한 RPN을 사용하였기 때문에 convolutional 연산이 공유되어 더 빠른 속도를 낼 수 있었다고 합니다.



![](.\faster rcnn_table1.png)

RPN과 Fast R-CNN 사이에서 convolutional layer를 공유했을때의 효과를 보여줍니다.

RPN+ZF의 경우가 mAP가 가장 높았습니다.

또한 proposal이 6000인 경우 mAP는 55.2%로 NMS가 mAP에 악영향을 미치지 않는 것을 알 수 있었다고 합니다.

![](.\faster rcnn_table2.png)

이번에는 RPN+VGG를 사용한 결과입니다.

ZF보다 VGG를 사용하였을때 생성된 proposal이 더 정확한 것을 확인할 수 있었습니다.

![](.\faster rcnn_table3.png)

시간을 비교한 table입니다.

VGG는 최종적으로 198ms가 소요되었으며 ZF는 59ms가 소요되었습니다.

또한 VGG에 convolutional feature가 공유되었을 경우에는 layer를 계산하는데 10ms가 걸린다고 하였습니다.

![](.\faster rcnn_table4.png)

scale과 ratio의 갯수를 다르게하여 적용한 결과입니다. 최종적으로는 3scale, 3ratio를 사용한 것이 mAP가 가장 높았습니다.

![](.\faster rcnn_table5.png)

대규모의 데이터가 어떤 영향을 미치는지에 대한 결과입니다.

결과를 보면 대규모 데이터가 성능을 향상시키는데에 매우 중요하다는 것을 확인할 수 있었습니다.

# Conclusion

RP를 생성하는데 RPN이 매우 효율적이고 정확하다는 것을 확인할 수 있었습니다.

또한 featrues를 공유합으로써 RP의 단계는 거의 cost-free에 가까워 졌다고 말하고 있습니다.