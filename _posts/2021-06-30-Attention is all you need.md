---
layout: post
title: Attention is all you need
data: 2021-06-30
excerpt: "Attention is all you need"
tags: [rnn, network]
coments: false
mathjax: true
---

# Attention is all you need

- 제목: Attention Is All You Need
- 저자: Ashish Vaswani 외 7명
- 인용수: 23,314회
- 학술지: nips2017

## Introduction

- RNN, LSTM, GRU는 sequence modeling에 많이 사용되었으며, 뛰어난 성능을 나타냄
- 하지만 이들의 sequential한 특성때문에 parallelization이 힘들뿐만 아니라, 길이가 길어 질 수록 메모리 부족문제가 나타남
- 따라서 이를 Transformer라는 attention mechanism만을 사용한 모델 구조를 제안하여 해결하고자 함

## Background

- sequential한 연산을 줄이고자 CNN을 기반으로한 Extended Neural GPU, ByteNet, ConvS2S 등 과 같은 모델들이 등장
- 하지만 이들은 input-output의 거리가 멀어질 수록 연산량은 증가(logarithmically)하였고, 장거리 의존성을 학습하는데 더 어렵게 만듬
- 반면에 Transformer는 상수 만큼의 연산을 요구
  - attention-weighted position을 평균냈기 때문에 effective resolution이 감소하였지만 이는 Multi-Head Attention을 통하여 해결

## Model Architecture

- 성능이 좋은 대부분의 sequence model은 encoder-decoder구조를 활용
- encoder
  - symbol representations $(x_1,...,x_n)$을 continuous representations $z=(z_1,...,z_n)$으로 바꿔 주는것
- decoder
  - z를 갖고 한 time에 한 원소씩 output sequence $(y_1,...,y_m)$을 생성(symbol)
- 모든 step은 auto-regressive이고, 다음 단계의 symbol을 생성할때 이전단계에서 생성된 symbol을 추가 입력
- Transformer는 encoder와 decoder에 self-attention과 point-wise, FC layer를 쌓아서 사용한 구조

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/attention/attention_fig1.png?raw=true)

### Encoder and Decoder Stacks

- Encoder:

  - N=6개의 동일한 layer로 stack하여 구성
  - 각 layer는 2개의 sub-layer를 갖고 있음
    - multi-head self-attention mechanism
    - position-wise fully connected feed-forward network
  - 각 sub-layer에 residual connection을 하고, 그 뒤에 normalization layer가 붙음
  - sub-layer의 출력은 $LayerNorm(x+Sublayer(x))$이며, $Sublayer(x)$는 sub-layer 자체로 구현되는 함수
  - residual connection을 용이하게 하기 위해 embedding layer를 포함한 모든 sub-layer는 $d_{model}=512$차원으로 고정

- Decoder:

  - Encoder와 마찬가지로 N=6개의 동일한 layer를 stack함

  - Encoder에서 2개의 sub-layer에다가 추가로 세번째 sub-layer를 추가

    - encoder stack의 output에 multi-head attention을 수행하는 layer

  - Encoder와 마찬가지로 residual connection을 한 후, normalization layer를 붙임

  - Decoder stack의 self-attention에서 출력을 생성할때, subsequent position에서 정보를 얻는것을 방지하기 위해 masking을 사용

    > 알고있는 단어로만 앞을 예측했다고 하기 위한 것
    >
    > 즉, 뒤에 오는 단어를 미리 예측하지 못하게
    >
    > 0~i-1번째 원소만 참조 하도록

### Attention

- Attention함수는 query와 key-value의 set을 통해 output으로 변환을 수행

  > query, key, value, output은 모두 벡터

- output은 value들의 weighted sum으로 계산

  > 이때 weight는 key에 대한 query의 compatibility function을 수행하여 얻어짐


#### Scaled Dot-Product Attention(compatibility function)

- input으로는 $d_k$차원의 query와 key, $d_v$차원의 value

- query와 모든 key를 dot product(내적)를 하고, 이를$\sqrt{d_k}$로 나눔(scaling 작업)

  > $d_k$가 커질수록 dot product가 커지며, softmax를 통과한 값이 0에 가까워짐
  >
  > 이를 방지하기 위한 scaling 작업

- 나눈 값을 softmax를 통과시켜 value에 곱해줄 weight를 구함

- 실제로는 여러 query와 그에따른 key, value를 matrix로 묶어 Q, K, V로 만들어 계산한다.

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/attention/attention_s1.png?raw=true)

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/attention/attention_fig2_1.png?raw=true)

#### Multi-Head Attention

- $d_{model}$의 벡터를 h로 나누어 $d_k,d_k,d_v$차원으로 linear projection한 다음 h번 학습 진행

  > 벡터의 크기를 줄이면서 병렬처리가 가능

- 이들을 병렬처리하여 $d_v$차원의 output(attention)을 concatenate 한 후, 최종 output을 도출

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/attention/attention_s2.png?raw=true)

- 본 논문에서는 h=8의 값을 사용
- $d_k,d_v=d_{model}/h=64$를 사용(512/8)

#### Applications of Attention in our Model

- Encoder-Decoder attention layer에서 query는 이전 Decoder layer에서 받음

- key와 value는 Encoder의 출력에서 받음

  > Decoder가 input의 모든 위치를 고려할 수 있도록 만들기 위함

- Encoder는 self-attention layer를 포함

- 여기서의 query, key, value는 Encoder의 이전 layer의 출력

  > Encoder는 이전 layer의 모든 위치를 고려할 수 있음

- Decoder에도 같은 self-attention layer가 있음

- 단, Auto-regressive 성질을 유지시켜주기 위해 출력을 생성할 때 다음 출력은 고려하지 않게 함

  > 현재 위치 외의 값은 예측하지 못하게 masking 작업

### Position-wise Feed-Forward Networks

- Encoder와 Decoder의 sub-layer는 FC feed-forward를 포함

  > position마다 독립적으로 적용 되기 때문에 position-wise

- 이는 두번의 linear transformation과 ReLU를 포함

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/attention/attention_s3.png?raw=true)

- input과 output은 동일하게 $d_{model}=512$, FFN내부의 hidden layer$(W_1\space등)$는 2048차원

### Embeddings and Softmax

- input output token을 $d_{model}$차원을 갖는 벡터로 만들어주는 learned embedding을 사용

- decoder의 output은 FC와 softmax를 거치기 때문에 next-token의 확률로 나옴

  > embedding layer와 softmax이전의 linear transformation에서 weight를 공유

- Embedding layer에는 weight에 $\sqrt{d_{model}}$을 곱해 줌

### Positional Encoding

- RNN과 CNN을 사용하지 않기때문에 위치에 대한 정보입력이 필요
- Encoder와 Decoder의 stack아래에 positional encoding 추가
- potitional encoding은 $d_{model}$ 차원과 동일하기 때문에 합치기 가능
- 본 논문에서는 sin/cos 함수를 사용

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/attention/attention_s4.png?raw=true)

> pos: 위치
>
> i: 차원

- 긴 sequence를 만나도 sin/cos 함수는 추정이 가능

## why Self-Attention

- 각 layer당 총 계산량이 줄어듬
- 동시에 병렬로 계산이 가능
- 멀리 떨어진 원소들간의 path length감소

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/attention/attention_table1.png?raw=true)

- table을 살펴보면 self-attention이 상수배로 작은 것을 확인 가능

## Training

- Dataset
  - German: WMT 2014 English-German dataset
  - French: WMT 2014 English-French dataset
- Batch size: 25,000
- Hardware: 8개의 P100GPU
- Schedule
  - Base Model: 12시간 = 10만 step * 0.4s/step
  - Big Model: 36시간 = 30만 step * 1.0s/step
- Optimizer: Adam
  - $\beta_1=0.9$
  - $\beta2=0.98$
  - $\epsilon=10^{-9}$
- learning_rate: 1 워밍업 단계에선 선형적으로 증가, 그 후 스텝 수의 역 제곱근에 비례하여 감소
  - warmup_steps = 4,000

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/attention/attention_s5.png?raw=true)

- Residual Dropout: $P{drop}=0.1$

## Result

![](https://github.com/jh79783/jh79783.github.io/blob/main/assets/img/attention/attention_table2.png?raw=true)