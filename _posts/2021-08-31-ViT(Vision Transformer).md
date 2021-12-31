---
layout: post
title: ViT(Vision Transformer)
data: 2021-08-31
excerpt: "Vision Transformer"
tags: [transformer]
coments: false
mathjax: true
---



# ViT(Vision Transformer)

- 제목: AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
- 저자: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov 외 9명 (Google Research, Brain Team)
- 인용수: 836회
- 학술지: ICLR 2021



![](C:\workspace\논문\vit\image1.png)



## Introduction

- 2017년에 Transformer가 공개된 후로 NLP에서는 주로 채택하여 사용 및 많은 SOTA를 달성하였으며 transformer의 엄청난 확장성덕에 100B(천억)개 이상의 parameter를 갖는 모델을 훈련하는 것이 가능해졌으며, 엄청난 크기의 모델과 데이터를 사용하더라도 saturation의 조짐이 보이지 않음

- 그러나 Computer Vision에서의 모델은 아직까지 CNN기반을 한 모델이 많음

- 따라서 본 논문은 transformer만을 이용해 computer vision에 적용시키고자 함

- 이를 위해 저자들은 이미지를 작은 patch(query)로 분할한 후 patch를 linear embedding하여 transformer의 input으로 사용함

  > patch는 추후 classification에서 사용할 token처럼 다루어짐

- ImageNet-21k을 사용하여 실험을 하였을 경우 기존 모델에 비해 정확도가 떨어졌으나, 14M~300M의 거대한 데이터를 사용한 경우 기존 모델보다 정확도가 능가하는 결과를 나타냄

## Related Work

- 지금까지 transformer를 vision에 적용시키고자 pixel끼리 계산, block으로 만들어 계산하는 등의 다양한 근사화한 방법을 사용하였음

- transformer의 특성으로 인해 pixel마다 계산을 하게 되면 연산량이 quadratic하게 증가하게 됨

  > e.g. image resolution: 250*250 = 62500 pixels -> 3906250000 calculations

## Method

![](C:\workspace\논문\vit\fig1.png)

- 연구진들은 기존 transformer의 이점을 이용하고자 최대한 original과 유사하게 설계하였음

  > NLP에서 사용한 확장방법을 그대로 사용하고자 함

### Vision Transformer(ViT)

- 원래 transformer는 1차원의 token embedding sequence를 input으로 받음

- 본 논문에서는 3차원 이미지 $H*W*C$를 $N*(P^2\cdot C)$로 각 patch에 대해 1D 텐서로 flatten함

  > H, W: image resolution
  >
  > C: image channel
  >
  > P \* P: patch resolution = patch의 H * W
  >
  > N: num patch ($N=\frac{HW}{P^2}$) -> sequence length(hyper parameter)

- positional embedding과 classification token vector를 추가해줌

  > positional embedding의 경우 픽셀마다가 아닌 patch의 position을 추가

- flatten한 patch를 linear projection을 통해 D차원으로 mapping함

  > size=$N*P^2*C$
  >
  > transformer의 hidden dim으로 변경하는 것(?)

- flatten된 patch들의 앞에 위에서 언급한 token vector를 맨 앞에 추가해 주어 output에 대한 representation vector의 역할을 수행함

- MLP로 이루어진 classification head가 붙어있는데, 이는 pre-train시 하나의 hidden layer를 갖는 MLP로 구현되며 fine-tuning시 단일 linear layer로 구현됨

- Encoder의 경우 MSA와 MLP로 구성되어 있으며 layernormalization의 경우 모든 patch에 적용됨
- Encoder의 MLP에서는 activation으로 ReLU가 아닌 GELU(Gaussian Error Linear Unit)를 사용하는 2개의 layer가 포함되어 있음(FC-GELU-FC)

> ![](C:\workspace\논문\vit\GELU.png)

![](C:\workspace\논문\vit\s1.png)

> $Z_0$: input
>
> $X_{class}$: classification token
>
> $X^n_p$: n번째 patch
>
> $E$: mapping된 vector(?)/linear projection(?)
>
> $Z^0_L$: L번째 Encoder layer의 0번째 token

### Fine-Tuning and Higher Resolution

- 저자들은 큰 데이터를 pre-train하고 작은 데이터를 fine-tuning하여 사용하였음
- 최근 연구에서 pre-train보다 fine-tuning시 더 높은 해상도로 하는것이 좋다는 것이 밝혀져 본 논문을 실험할 때도 더 높은 해상도에서 fine-tuning을 진행하였음
- 단, 이때 $P*P$의 patch resolution은 유지하며, sequence를 늘리는 방향으로 조절함



## Experiments

- pre-train으로는 ImageNet, ImageNet-21k, JFT-300M을 사용
- fine-tuning으로는 pre-train에 사용하고 남은 ImageNet, ImageNet ReaL, CIFAR-10/100 등을 사용

![](C:\workspace\논문\vit\table1.png)

### Training & Fine-tuning

- pre-train
  - optimizer: Adam
    - $\beta_1$=0.9
    - $\beta_2$=0.999
  - batch_size=4096
- fine-tuning
  - optimizer: SGD
  - batch_size=512

![](C:\workspace\논문\vit\table2.png)

- 기존의 SOTA는 BiT-L/Noisy Student였으나 모든 dataset에 대해 ViT가 능가하는 것을 확인할 수 있으며, 비용 또한 더 작은 것을 확인 가능
- 저자들은 본 실험을 진행하며 pre-train에서 training schedule이나 optimizer등의 다른 파라미터가 영향을 준다는 것에도 주목함

![](C:\workspace\논문\vit\fig5.png)

- training cost를 평가하는데 ViT가 성능/계산코스트에서 더 좋은 지표를 나타내었음
- 하지만 hybrid model의 경우 계산량이 적은 부분에서는 더 뛰어남

## Conclusion

- 저자들은 기존의 다른 모델과 다르게 original과 최대한 비슷한 transformer를 통해 computer vision에 적용하였음
- 모든 데이터셋에 대해 기존 SOTA모델들을 뛰어넘으며 compute cost까지 작지만, 큰 규모의 데이터셋이 요구됨
- 저자들은 다음과 같은 도전과제를 남김
  1. detection이나 segmentation에서의 적용
  2. self-supervised pre-training 향상
  3. 성능 향상을 위한 ViT의 확장

