---
layout: post
title: GoogLeNet
data: 2021-01-26
excerpt: "GoogLeNet"
tags: [GoogLeNet, network]
coments: false
mathjax: true
---



# GoogLeNet

이 논문은 CVPR2015에 공개 되었으며, 제목은 Going deeper with convolutions입니다. 

ILSVRC14에서 classification과 detection을 위한 Inception이라 불리는 새로운 아키텍처를 제안하였으며, 이 방법을 통해 state-of-the-art성능을 달성하였습니다.

이 아키텍처의 주된 특징은 컴퓨팅 리소스의 활용을 향상시켰다는 점입니다. 또한 연산 비용을 일정하게 유지하면서 네트워크의 깊이와 너비를 증가시켰습니다.

ILSVRC14에 제출한 모델은 22layer의 deep network이고, 이를 GoogLeNet이라 칭하였습니다.

ILSVRC14에 제출한 GoogLeNet은 AlexNet보다 12배 적은 parameter를 사용하였음에도 훨씬 좋은 성능을 보였습니다. 또한 objet detection에서 R-CNN과 같은 알고리즘을 사용해 더 큰 성능 향상이 나타났습니다.

대부분의 실험에서 컴퓨터의 연산 예산을 15억번을 넘지 않도록 하였으며, 이것은 단순히 학술적 호기심이아닌 실제로 사용할 수 있도록 설계되었다고 합니다.

본 논문에서는 컴퓨터 비전에서 효율적인 deep neural network 구조에 초점을 맞추었습니다. 이를 Inception이라고 부릅니다.

deep이라는 단어에는 두가지의 다른 의미를 사용하였습니다.

첫번째로 Inception moudule이라는 새로운 구조를 도입한다는 의미, 두번째로 네트워크의 깊이를 증가시킨다는 의미입니다.

## Related Work

영장류의 시각 피질에 대한 신경 과학에서 영감을 얻은 연구에서는 multiple scale을 다루기 위해 크기가 다른 fixed Garbor filter를 사용하였습니다. 본 논문에서도 비슷한 전략을 사용하지만, Inception구조의 모든 filter가 학습한다는 차이점이 있습니다. 또한, GoogLeNet의 경우는 Inception layer가 여러번 반복되어 22-layer deep model로 나타납니다.

Network-in-Network는 신경망의 표현력을 증가시키기 위한 접근법입니다.

이 모델에서는 1\*1 convolutional layer를 추가적으로 사용하여 depth를 증가시켰습니다.

1\*1 convolution은 두개의 목적을 갖고있습니다.

1. 컴퓨터의 병목현상을 제거하기 위한 차원축소모듈
2. 네트워크의 크기 제한

이를통해서 성능 저하없이 depth와 width를 늘릴 수 있다고 합니다.

## Motivation and High Level Considerations

neural network의 성능을 향상시키는데 가장 직접적인 방법은 depth나 width를 늘리는 것인데, 특히 엄청난 양의 label된 training data가 있을경우 이 방법이 쉽고 안전하게 고성능의 모델을 학습시키는 방법입니다. 그러나 이 경우 두개의 중요한 단점이 존재합니다.

크기가 커질수록 더 많은 parameter를 갖는다는 것을 의미하고, 이것은 overfitting이 되는 쉬운 경향이 있다는 것을 말합니다. 특히 labeled된 training set이 제한적인경우 이것은 더욱 심해진다고 합니다. 

또 하나의 단점은 네트워크의 크기가 증가하면 컴퓨터의 자원사용량이 극도로 증가하게 되는점입니다. 

첫번째 단점의 예로, ImageNet의 dataset에서 시베리안 허스키와 에스키모 개를 구분하는 것처럼 다양한 클래스처럼 세분화된 경우 생기는 병목현상 입니다.

두번째 단점의 예로는 conv layer의 filter가 증가하면 계산량이 제곱이 되고, 이떄 추가된 filter의 weight는 대부분 0에 가까워지게되는 컴퓨터 자원사용의 낭비입니다. 따라서 컴퓨터의 자원을 효율적으로 분배하는 것이 중요하다고 설명하고 있습니다.

두 단점을 해결하기 위해서 sparsity를 도입하고, convolution안에서 FC layer를 sparse connected로 변경하는 것입니다. 

> 즉, sparse한 것으로 변경한다는 것은 직접 선택 혹은 통계적 분석을 통해 간접 선택을하거나, dropout같은 방법을 사용하거나, 학습 중 자연스럽게 발생한 deactivated node를 다음 layer의 연산에 포함되지 않도록 sparse data struture로 바꾸는 방법으로 생각할 수 있습니다.

LeNet에서도 convolution안에서 sparse하게 사용하였습니다.

![](..\lenet\LeNet-5_table1.jpg)

하지만 오늘날의 컴퓨터의 구조는 non-uniform sparse한 data구조를 계산하기엔 너무 비효율적이기 때문에 sparse한 matrix를 densely한 submatrix로 clustering하는 방법을 제안하였으며, 저자들은 이 방법이 non-uniform한 deeplearning 아키텍처의 자동화 기법에 비슷한 방법이 멀지않은 미래에 활용될 수 있을것이라 생각하였습니다.

이 Inception 구조는 sparse 구조를 근사화 하면서도 dense하고 쉽게 이용할 수 있도록 설계된 network topology construction 알고리즘을 평가하기 위해서 연구가 시작되었다고 합니다. 연구를 통해서 Inception구조는 R-CNN과 scalabel object detection에서 base network로 사용되었을때, locailzation과 object detection분야에서 성능이 향상되었다고 합니다. 하지만 이것이 의도한 것대로 이끌어졌는지에 대해 의문이 남아있어 더 많은 연구가 필요하다고 말하고 있습니다.

## Architectural Details

Inception 구조의 main idea는 convolutional vision network에서 optimal local sparse structure의 근사화와 dense한 요소들을 쉽게 이용가능하게 구성하는 것입니다.

translation invariance를 가정하는데요 이것은 convolutional building block으로 network가 구성되는 것을 의미합니다.

> translation invariance: 이미지 내의 어떤 특징이 평행 이동 되어도 활성화 여부가 달라지지 않는 성질

이를 위해 optimal local construction을 찾고 이것을 공간상에서 반복하는 것이 요구됩니다.

본 논문에서는 이전 layer의 unit은 입력 이미지의 일부 영역에 해당하고, 이런 unit들은 filter bank로 그룹화 된다고 가정합니다. 따라서 lower layer에서는 관련된unit들이 local region에 집중됩니다. 즉, 하나의 region에 더많은 cluster들이 집중되고, 이것은 다음 layer에서 1\*1의 convoltuion 으로 작동할 수 있다 말하고있습니다.

patch alignment issues(패치 정렬 문제)를 피하기위해 Inception구조에서 filter의 크기를 1\*1, 3\*3, 5\*5로 제한하였습니다. 이러한 크기로 제한한 이유로는 단순히 편리하기 때문에 크기를 제한하였지 필수적으로 filter의 크기를 똑같이 사용해야 하는 것은 아닙니다.

> 짝수인 경우 patch의 중심을 어디로 해야할지 생각해야한다.

또한 현재 성공적인 convolutional network에서 pooling연산이 필수라는 것을 확인할 수 있다고 합니다. 따라서 각각 단계에서 다른 병렬 pooling path를 추가하게되면 추가적인 이득을 얻을 수 있다고 설명하고 있습니다.



<img src="./googlenet_module.png" style="zoom:75%;" />

이런 Inception module은 서로의 위에 쌓이게 되는데, 이것은 출력되는 correlation statistics가 달라질 수 있습니다. 즉, high layer에서 더 추상적인 특징들이 추출되기 때문에 공간적인 집중도가 감소할 것으로 예상이 된다고 합니다. 이는 high layer로 갈수록 3\*3, 5\*5와 같은 비율이  증가하는 것을 의미합니다.

사진의 a와 같은 module의 경우 5\*5 convoltuion을 아무리 적게 사용한다 하여도 filter가 많아지면 cost가 비싸질 수 있는 문제가 존재합니다. 여기에 pooling까지 추가된다면 이런 문제는 더욱 크게 나타나게 됩니다.

> convolution 결과를 concatenation으로 연결하기 때문에 output의 채널 수가 많아지게 됩니다.
>
> 여기에 output의 채널 수가 input과 같은 pooling이 추가된다면 채널 수가 2배 이상 누적되기 때문에 치명적이라고 설명하고 있습니다.

즉, optimal sparse tructure를 잘 처리가 가능함에도 불구하고 매우 비효율적으로 수행되기 때문에 갑자기 계산량이 증가하는 문제가 있습니다.

Inception 구조에서는 이 문제를 해결하기 위해 계산량이 너무 많을때 차원을 효율적으로 줄이는 방법을 제안하였습니다.

저차원에서는 상대적으로 큰 image patch에 대한 정보를 포함 할 수 있지만, 이것은 압축된 형태로 dense하게 나타나며 이렇게 압축된 정보는 처리하기가 어렵다는 단점이 있습니다. 이러한 정보는 대부분 sparse하게 유지되어야 하는데, 필요하다면 압축을 해야하는 경우도 생기게 됩니다. 따라서 3\*3, 5\*5 convolution을 수행하기 전 연산량을 줄이기위해 1\*1 convolution을 수행하였습니다. 이때, 감소역활 이외에도 이중 목적을 위해 ReLU를 사용하였습니다.

> 연산량 감소와 non-linearity를 얻기위한 목적
>
> 1\*1convolution을 사용한 이유: convolution을 사용하게 되면 채널 수 조절이 가능한데, 1\*1을 사용함으로 써 값은 그대로 유지하되 채널의 수만 감소시키기 위해서

Inception network는 이러한 module들이 쌓여 구성된 network입니다. 가끔씩 feature map을 줄이기 위해 stride가 2인 max-pooling을 사용하였으며, 메모리의 효율성을 위하여 하위 layer에서는 전통적인 convolution을 유지하였고, 상위 layer에서는 Inception module을 사용하는 것이 좋다고 판단하였습니다. 이것은 효율을 고려한 것이며 필수적인 것은 아니라고 추가적으로 설명하고 있습니다.

Inception 구조는 계산을 효율적으로 하기 때문에 성능이 떨어지더라도 저렴하게 깊고 넓은 network를 구축할 수 있다고 말하고 있습니다.

## GoogLeNet

ILSVRC 2014에 GoogLeNet이라는 팀으로 출전하였으며, Inception구조의 형태를 가르키는 것으로도 사용하였습니다.

네트워크의 세부적인 설명은 생략한다고 하는데요. 그 이유로는 저자들이 GoogLeNet을 갖고 실험을 하였을때, parameter의 영향이 매우 작았기 때문이라고 설명하고 있습니다.

가장 성공적인 모델을 table에 적었으며, 추가적으로 구성한 모델은 6개가 더있습니다. 따라서 총 7개의 모델이 존재합니다.

![](./googlenet_table1.png)

> 3\*3 reduce, 5\*5 reduce는 3\*3, 5\*5크기의 convolution을 적용하기 전, reduction하는 1\*1filter 수를 의미한다고 합니다.
>
> 첫번째 inception 모듈을 보면(inception(3a))입력되는 이미지의 크기는 28\*28\*256입니다. 이것을 3\*3reduce를 통해 96채널로 줄인뒤 3\*3 convolution을 통해 128채널의 output이 나오게 됩니다.
>
> inception 모듈을 사용하는 최초의 시점에서 시각적인 이미지는 사라지고 특징 벡터로 작고 두껍게 변환되었다고 합니다.
>
> FC에서는 이미 입력데이터가 1\*1\*1024로 height와 width가 1\*1로 되어있어 학습할 파라미터가 1000k인 것을 확인할 수 있습니다.
>
> > AlexNet의 경우 4096\*4096으로 되어있음.

Inception 모듈 내부에서 모든 convolution에서 ReLU를 사용합니다.

이 네트워크는 연산을 효율적이고 실용적으로 설계되었다고 합니다. 따라서 계산 자원에 한계가 있으며, 메모리가 작은 장치에서도 실행이 가능하다고 합니다.

파라미터가 있는 layer만 계산한다면, 22개의 layer를 갖고있고, pooling을 포함한다면 27개의 layer가 존재합니다.

> pooling layer에는 학습할 파라미터가 없다.

마지막에 Linear layer(FC layer)를 사용했는데, 이것은 다른 label set에 쉽게 적용할 수 있는 편의 때문에 사용하였다고 합니다. 또한 FC 대신 average pooling을 사용을 통해 top-1 accuracy가 0.6% 향상된 것을 확인 할 수 있었다고 합니다. 하지만 그럼에도 여전히 dropout을 필수적으로 사용해야 한다고 저자는 말하고 있습니다.

저자들은 Network에 보조분류기를 중간 layer에 추가해주어서 regularization효과와 vanishing gradient 문제를 해결해 줄것으로 생각하였습니다. 보조분류기를 Inception 모듈의 4a, 4d의 출력에서 나온 작은 convolution network의 형태를 갖고있으며, 학습중에는 이들의 loss에 weight를 적용하여 network의 최종 loss에 추가하게 됩니다.  

> 보조분류기는 inference 시에는 제거합니다.(validation 단계?)
>
> 보조분류기의 loss 30%만 고려합니다.

보조분류기를 통하여 약 0.5%의 성능 향상이 나타났다고 합니다.

따라서 보조분류기를 통한 네트워크에 대한 정확한 구조는 다음과 같습니다.

- 5\*5크기의 average pooling layer, stride는 3, 출력 shape은 4a에서 4\*4\*512, 4d에서 4\*4\*528
- 1\*1 conv layer, ReLU
- FC layer(1024 노드), ReLU
- dropout = 0.7
- linear layer(FC layer)에 softmax를 사용한 1000-class classifier

![](.\googlenet_.png)

## Training Methodology

GoogLeNet은 최신의 모델과 데이터를 병렬을 이용하여 학습시키는 DistBlief를 사용하여 학습하였습니다. 또한 학습은 CPU기반으로만 진행하였는데, 몇개의 high-end GPU를 사용한다면 1주일 내로 학습이 완료될 것으로 추측하고 있습니다.

학습에서는 momentum을 0.9로 설정하였고, learning rate schedule은 8 epoch마다 4% 감소하도록 적용하였습니다.

이미지를 샘플링하는방법에는 ILSVRC 2014까지 크게 변화가 되었다고합니다. 이미 끝난 모델들은 다른 옵션을 사용해서 또 학습을 하였고, 때때로는 dropout, learning rate등의 parameter를 변경하여 학습하기도 하였기 때문에 가장 효과적인 방법을 제공하기는 어렵다고 말하고 있습니다.

대회 이후에는 종횡비를 3/4 혹은 4/3으로 제한하여 8% \~ 100%의 크기로 patch sampling하는 것이 매우 잘 작동한다는 것을 발견 하였으며, photometric distortion(사진왜곡)이 overfitting 방지에 유용하다는 것을 발견하였습니다.

## ILSVRC 2014 Classfication Challenge Setup and Results

classfication challenge는 이미지를 ImageNet의 1000개의 카테고리 중 하나로 분류하는 작업을 의미합니다.

성능평가로는 두 종류의 수치를 보게 됩니다.

첫번째로 top-1 accuracy rate이고 둘째로는 top-5 error rate입니다. 이 대회에서는 top-5 error rate로 ranking을 결정하였습니다.

GoogLeNet은 추가적인 데이터를 학습에 사용하지 않았으며, 테스트 과정에서는 side를 256/288/320/352 4가지 scale로 크기를 조정한 후 왼쪽, 가운데, 오른쪽을 잘라줍니다. 이때 자른 이미지에 대해서 모서리와 중앙에 224\*224크기로 crop하고, 자른 이미지를 224\*224로 resize한 것과 미러링된 버전을 취해주는 crop방식을 사용하였다고 합니다. 이는 AlexNet보다 더 적극적인 crop방식이라고 소개하고 있습니다.

![](.\googlenet_classification_result.png)

위의 테이블을 살펴보면 GoogLeNet은 top-5 error를 6.67%로 1위를 차지하였습니다.

## ILSVRC 2014 Detection Challenge Setup and Results

detection분야는 bounding box를 생성하는 것입니다. detect된 object가 gt class와 일치하고 box가 50%이상 되는경우 정답으로 계산하였습니다. 관련이 없는 검출대상에 대해서는 false positive로 간주하여 패널티를 받았습니다.

GoogLeNet이 검출을 위해 사용한 접근법은 R-CNN과 비슷하지만 region classifier로 inception 모델을 사용하여 보강하였습니다.

region에 대한 classification에는 6개의 GoogLeNet을 앙상블하여 정확도가 40%에서 43.9%로 향상된 결과를 나타내었습니다.

![](.\googlenet_detection_result.png)

위의 table을 detection 결과를 나타냅니다.

2013년에 비해 2배 정도로 향상된 것을 볼 수 있습니다.