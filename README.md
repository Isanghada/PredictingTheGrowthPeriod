# 🔎딥 러닝 프로젝트 : 생육 기간 예측
## Introduction
- 한 쌍의 이미지를 입력받아 작물의 생육 기간을 예측을 목적으로 진행한 프로젝트입니다.(<a href='https://dacon.io/competitions/official/235851/overview/description' target='_blink'>데이콘 경진 대회</a>)
- 이미지를 CNN을 통해 분석하고 생육 기간 예측을 진행한다.
  
## Contents
- [1. 프로젝트 소개](#1-프로젝트-소개) 
- [2. 데이터 전처리](#2-데이터-전처리)
- [3. 모델링](#3-모델링)
  - [Lenet](#lenet)
  - [Mobilenet_v2](#mobilenet_v2)
  - [Resnext50_32x4d](#resnext50_32x4d)
- [4. 최종 결과](#4-최종-결과)
- [5. 한계 및 보완점](#5-한계-및-보완점)
- [6. 참고 자료](#6-참고-자료)
- [7. 구성 인원](#7-구성-인원)

## 1. 프로젝트 소개
### 배경
- 스마트팜, IT 기술을 동원하여 더욱 효율적인 작물 재배의 가능성
- 작물의 효율적인 생육을 위한 가장 최적의 환경 도출의 일환으로 식물의 이미지를 이용해 성장 기간 예측
- <a href='https://dacon.io/competitions/official/235851/overview/description' target='_blink'>데이콘 경진 대회</a>

### 프로젝트 개요
- 구성인원 : 김남규, 박이정, 유현준
- 수행기간 : 2021년 12월 20일 ~ 2021년 12월 30일
- 목표 : 한쌍의 이미지를 통한 생육 기간 예측
- 데이터 : 적상추, 청경채 이미지
  - 학습 : 753개(청경채 353개, 적상추 400개)
  - 테스트 : 307개(청경채 139개, 적상추 168개)
  - 평가지표 : RMSE(Root Mean Square Error)

### 개발 환경
- 시스템 : Colab
- 언어 : Python
- 라이브러리 : **Pandas**, **Numpy**, **PIL**, **Torch**, **Torchvision**
- 알고리즘 : CNN(LeNet, Mobilenet_v2, Resnext50_32x4d)

## 2. 데이터 전처리
#### ▪이미지 Resize
- 원본 사이즈 : (3280, 2464)
  - 시스템 사양의 한계로 Colab 활용
  - GPU 메모리 초과 또는 Colab 사용 시간 초과 발생
  - 이미지 사이즈를 미리 조정하여 해결
- 사용 사이즈 : (224, 224)
  - 이미지넷을 활용한 사전 학습 모델(Mobilenet_v2, Resnext50_32xd4) 사용을 위해 해당 사이즈로 조절
  - PIL 라이브러리를 활용해 모든 이미지 사이즈 조절
  ```python
  import os
  from PIL import Image

  def data_resize(species_nm,root_path):
    os.mkdir(root_path +'/'+ species_nm + '_resize')
    # 서브 폴더 생성
    for sub_path in os.listdir(root_path +'/'+ species_nm): 
      os.mkdir(root_path +'/'+ species_nm + '_resize/' + sub_path)
      # 이미지 resize 및 저장
      for image_path in glob(root_path +'/'+ species_nm + '/' + sub_path + '/*'): 
        image_file_name = image_path.split('/')[-1]
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img.save(root_path +'/'+ species_nm + '_resize/' + sub_path + '/' + image_file_name)
  ```

## 3. 모델링
<p align="center">
<img src = 'https://postfiles.pstatic.net/MjAyMjAzMTVfMjMz/MDAxNjQ3MzE5MzI5MjI4.CAxfbx6lZKslVNr3rxU0zRiijUd_9tDXNvf_nEl7or8g._MuxaQOaULAjEVv8QOkZRfZbaCXyLMzQE92DdZz_NAog.PNG.isanghada03/SE-13d56f79-5ff1-458f-8e2c-8b422cdb29ef.png?type=w966' width="50%" height="50%"></p>  

- 모델 구조
  - after_net, before_net : CNN을 통해 이미지에서 생육 기간을 도출하는 모델
    - [Lenet, Mobilenet_v2, Resnext50_32xd4]으로 구성
  - CompareNet : 각 이미지에서 도출한 결과로 두 이미지 간의 생육 기간을 도출하는 모델

### Lenet
<p align="center"><img src='https://postfiles.pstatic.net/MjAyMjAzMTVfMTM1/MDAxNjQ3MzE3NzA0ODc2.FMveD6zMA6VkEA-KodxsB6IMcIvYn2DPesq57NIKNqcg.YFUxzbJwiSPww9Gyf6EzP5Xkrc42nyS9HNbIH8iXuQkg.PNG.isanghada03/image.png?type=w966'></p>

- 가장 기본적인 CNN(Convolutional Neural Network)으로 Layer 입력값과 출력값을 이미지 크기에 맞게 수정
- 간단한 모델로 동작 과정을 살펴보기 위해 진행

```python
import torch
from torch import nn
class LeNet(nn.Module):

  def __init__(self):
    super(LeNet, self).__init__()
    self.feature_extractor1 = nn.Sequential( 
      nn.Conv2d(3, 6, kernel_size=5, padding=2), 
      nn.ReLU(),
      nn.AvgPool2d(kernel_size=2, stride=2))
    self.feature_extractor2 = nn.Sequential(
      nn.Conv2d(6, 16, kernel_size=5), 
      nn.ReLU(),
      nn.AvgPool2d(kernel_size=2, stride=2))
    
    self.feature_extractor3 = nn.Sequential(
      nn.Flatten(),
    )
    self.feature_extractor4 = nn.Sequential(
      nn.Linear(16*54*54, 120), 
      nn.ReLU(),  
    )
    self.feature_extractor5 = nn.Sequential(
      nn.Linear(120, 84), 
      nn.ReLU(),
    )
    self.feature_extractor6 = nn.Sequential(
      nn.Linear(84, 10)
    )  

    self.output = nn.Sequential(
      nn.Linear(10, 1),
      )


  def forward(self, x):
    x = self.feature_extractor1(x)
    print("x1",x.shape)
    x = self.feature_extractor2(x)
    print("x2",x.shape)
    x = self.feature_extractor3(x)
    print("x3",x.shape)
    x = self.feature_extractor4(x)
    print("x4",x.shape)
    x = self.feature_extractor5(x)
    print("x5",x.shape)
    x = self.feature_extractor6(x)
    print("x6",x.shape)
    logits = self.output(x)
    print("logits", logits.shape)
    return logits

class CompareNet(nn.Module):

    def __init__(self):
        super(CompareNet, self).__init__()
        self.before_net = LeNet()
        self.after_net = LeNet()

    def forward(self, before_input, after_input):
        before = self.before_net(before_input)
        after = self.after_net(after_input)
        delta = before - after
        return delta
```  
### Mobilenet_v2
- 사전 학습 모델을 사용해 구현한 모델을 성능을 높이고자 시도
- 가벼우면서 성능이 좋다고 알려진 Mobilenet_v2 사용
- EPOCHS 변경, Image Augmentation 등 다양한 방법을 통해 성능 향상 시도
```python
import torch
from torch import nn
from torchvision.models import mobilenet_v2

class CompareCNN(nn.Module):
  def __init__(self):
    super(CompareCNN, self).__init__()
    self.mobile_net = mobilenet_v2(pretrained=True)
    self.fc_layer = nn.Linear(1000, 1)

  def forward(self, input):
    x = self.mobile_net(input)
    output = self.fc_layer(x)
    return output

class CompareNet(nn.Module):
  def __init__(self):
    super(CompareNet, self).__init__()
    self.before_net = CompareCNN()
    self.after_net = CompareCNN()

  def forward(self, before_input, after_input):
    before = self.before_net(before_input)
    after = self.after_net(after_input)
    delta = before - after
    return delta
```

### Resnext50_32x4d
- Mobilenet_v2보다 복잡하고 성능이 좋은 모델로 변경하여 비교 진행
```python
import torch
from torch import nn
from torchvision.models import resnext50_32x4d

class CompareCNN(nn.Module):
  def __init__(self):
    super(CompareCNN, self).__init__()
    self.resnext50 = resnext50_32x4d(pretrained=True)
    self.fc_layer = nn.Linear(1000, 1)

  def forward(self, input):
    x = self.resnext50(input)
    output = self.fc_layer(x)
    return output

class CompareNet(nn.Module):
  def __init__(self):
    super(CompareNet, self).__init__()
    self.before_net = CompareCNN()
    self.after_net = CompareCNN()

  def forward(self, before_input, after_input):
    before = self.before_net(before_input)
    after = self.after_net(after_input)
    delta = before - after
    return delta
```

## 4. 최종 결과
- 데이콘 제출 결과
  - public : 전체 테스트 데이터 중 25%
  - private : 전체 테스트 데이터 중 75%

<p align='center'><img src='https://postfiles.pstatic.net/MjAyMjAzMTVfMjQg/MDAxNjQ3MzIyMjk0OTI1.bWLSCrZi-tHPAOomuuvA1MsO4vlG3x6bWdG-IOzm-L0g.WWasJJvFlqFtzPA4ZQg_Sh1GoH5HHxdPzCTf9FAM56Ig.PNG.isanghada03/%EA%B7%B8%EB%A6%BC2.png?type=w966' width="90%" height="90%"></p>

- Feature Map
  - CNN이 이미지를 분석하는 방식을 살펴보고자 진행
  - Lenet과 Mobilenet_v2로 진행
  - [Lenet]  
  <img src='https://postfiles.pstatic.net/MjAyMjAzMTVfODUg/MDAxNjQ3MzIzMzM2Nzkw.cYt21LFNY90aL2X_3IOKpnDAygkuPJdRm-R3hxo21Cgg.Cq25sAMBdFiaiEXJjxtNwsC92WCXsySdQzyFcRp_E4Qg.PNG.isanghada03/%EA%B7%B8%EB%A6%BC3.png?type=w966' width='70%' height='70%'>
  
  - [Mobilenet_v2]  
  <img src='https://postfiles.pstatic.net/MjAyMjAzMTVfMjEy/MDAxNjQ3MzIzMzM2NjMy.UxrVQfGGRdTg7H4hGVSVBEIcN7M5DsuVbTtzn6cQcxYg.BoUPIuaN7mXmfaKXi-71xU3qZN9R2ZoTTpkXcgCtRbQg.PNG.isanghada03/%EA%B7%B8%EB%A6%BC4.png?type=w966' width='70%' height='70%'>  
  
  - **주로 잎이 강조되는 것 확인**

## 5. 한계 및 보완점
#### 🛠시스템 사양 부족
- 사용 데이터가 이미지이고 사전 학습 모델의 Layer가 높은 시스템 사양이 필요했다.
- Colab을 통해 진행할 수는 있었지만, 주로 GPU 메모리 이슈가 발생하여 진행이 막힌 경우가 있었다.
  - 사양 부족과 학습 시간이 오래 걸려 주로 epoch을 변경하며 비교하였다.
- 사양이나 학습 시간 문제가 해결이 된다면 최적의 조건을 탐색하고 여러 모델들을 앙상블하면 조금 더 완성도 높은 모델이 될 수 있을 것이라 생각한다.

## 6. 참고 자료
- 데이콘 코드 : https://dacon.io/competitions/official/235851/codeshare
- mobilenet_v2 : https://pytorch.org/vision/main/generated/torchvision.models.mobilenet_v2.html
- resnext50_32x4d : https://pytorch.org/vision/main/generated/torchvision.models.resnext50_32x4d.html

## 7. 구성 인원
- 김남규 (<a href = 'https://github.com/Isanghada' href='_blank'>github</a>)
- 박이정 (<a href = 'https://github.com/happyfranc' href='_blank'>github</a>)
- 유현준 (<a href = 'https://github.com/hyunjuyo' href='_blank'>github</a>)
