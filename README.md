# ๐๋ฅ ๋ฌ๋ ํ๋ก์ ํธ : ์์ก ๊ธฐ๊ฐ ์์ธก
## Introduction
- ํ ์์ ์ด๋ฏธ์ง๋ฅผ ์๋ ฅ๋ฐ์ ์๋ฌผ์ ์์ก ๊ธฐ๊ฐ์ ์์ธก์ ๋ชฉ์ ์ผ๋ก ์งํํ ํ๋ก์ ํธ์๋๋ค.(<a href='https://dacon.io/competitions/official/235851/overview/description' target='_blink'>๋ฐ์ด์ฝ ๊ฒฝ์ง ๋ํ</a>)
- ์ด๋ฏธ์ง๋ฅผ CNN์ ํตํด ๋ถ์ํ๊ณ  ์์ก ๊ธฐ๊ฐ ์์ธก์ ์งํํ๋ค.
  
## Contents
- [1. ํ๋ก์ ํธ ์๊ฐ](#1-ํ๋ก์ ํธ-์๊ฐ) 
- [2. ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ](#2-๋ฐ์ดํฐ-์ ์ฒ๋ฆฌ)
- [3. ๋ชจ๋ธ๋ง](#3-๋ชจ๋ธ๋ง)
  - [Lenet](#lenet)
  - [Mobilenet_v2](#mobilenet_v2)
  - [Resnext50_32x4d](#resnext50_32x4d)
- [4. ์ต์ข ๊ฒฐ๊ณผ](#4-์ต์ข-๊ฒฐ๊ณผ)
- [5. ํ๊ณ ๋ฐ ๋ณด์์ ](#5-ํ๊ณ-๋ฐ-๋ณด์์ )
- [6. ์ฐธ๊ณ  ์๋ฃ](#6-์ฐธ๊ณ -์๋ฃ)
- [7. ๊ตฌ์ฑ ์ธ์](#7-๊ตฌ์ฑ-์ธ์)

## 1. ํ๋ก์ ํธ ์๊ฐ
### ๋ฐฐ๊ฒฝ
- ์ค๋งํธํ, IT ๊ธฐ์ ์ ๋์ํ์ฌ ๋์ฑ ํจ์จ์ ์ธ ์๋ฌผ ์ฌ๋ฐฐ์ ๊ฐ๋ฅ์ฑ
- ์๋ฌผ์ ํจ์จ์ ์ธ ์์ก์ ์ํ ๊ฐ์ฅ ์ต์ ์ ํ๊ฒฝ ๋์ถ์ ์ผํ์ผ๋ก ์๋ฌผ์ ์ด๋ฏธ์ง๋ฅผ ์ด์ฉํด ์ฑ์ฅ ๊ธฐ๊ฐ ์์ธก
- <a href='https://dacon.io/competitions/official/235851/overview/description' target='_blink'>๋ฐ์ด์ฝ ๊ฒฝ์ง ๋ํ</a>

### ํ๋ก์ ํธ ๊ฐ์
- ๊ตฌ์ฑ์ธ์ : ๊น๋จ๊ท, ๋ฐ์ด์ , ์ ํ์ค
- ์ํ๊ธฐ๊ฐ : 2021๋ 12์ 20์ผ ~ 2021๋ 12์ 30์ผ
- ๋ชฉํ : ํ์์ ์ด๋ฏธ์ง๋ฅผ ํตํ ์์ก ๊ธฐ๊ฐ ์์ธก
- ๋ฐ์ดํฐ : ์ ์์ถ, ์ฒญ๊ฒฝ์ฑ ์ด๋ฏธ์ง
  - ํ์ต : 753๊ฐ(์ฒญ๊ฒฝ์ฑ 353๊ฐ, ์ ์์ถ 400๊ฐ)
  - ํ์คํธ : 307๊ฐ(์ฒญ๊ฒฝ์ฑ 139๊ฐ, ์ ์์ถ 168๊ฐ)
  - ํ๊ฐ์งํ : RMSE(Root Mean Square Error)

### ๊ฐ๋ฐ ํ๊ฒฝ
- ์์คํ : Colab
- ์ธ์ด : Python
- ๋ผ์ด๋ธ๋ฌ๋ฆฌ : **Pandas**, **Numpy**, **PIL**, **Torch**, **Torchvision**
- ์๊ณ ๋ฆฌ์ฆ : CNN(LeNet, Mobilenet_v2, Resnext50_32x4d)

## 2. ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ
#### โช์ด๋ฏธ์ง Resize
- ์๋ณธ ์ฌ์ด์ฆ : (3280, 2464)
  - ์์คํ ์ฌ์์ ํ๊ณ๋ก Colab ํ์ฉ
  - GPU ๋ฉ๋ชจ๋ฆฌ ์ด๊ณผ ๋๋ Colab ์ฌ์ฉ ์๊ฐ ์ด๊ณผ ๋ฐ์
  - ์ด๋ฏธ์ง ์ฌ์ด์ฆ๋ฅผ ๋ฏธ๋ฆฌ ์กฐ์ ํ์ฌ ํด๊ฒฐ
- ์ฌ์ฉ ์ฌ์ด์ฆ : (224, 224)
  - ์ด๋ฏธ์ง๋ท์ ํ์ฉํ ์ฌ์  ํ์ต ๋ชจ๋ธ(Mobilenet_v2, Resnext50_32xd4) ์ฌ์ฉ์ ์ํด ํด๋น ์ฌ์ด์ฆ๋ก ์กฐ์ 
  - PIL ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ํ์ฉํด ๋ชจ๋  ์ด๋ฏธ์ง ์ฌ์ด์ฆ ์กฐ์ 
  ```python
  import os
  from PIL import Image

  def data_resize(species_nm,root_path):
    os.mkdir(root_path +'/'+ species_nm + '_resize')
    # ์๋ธ ํด๋ ์์ฑ
    for sub_path in os.listdir(root_path +'/'+ species_nm): 
      os.mkdir(root_path +'/'+ species_nm + '_resize/' + sub_path)
      # ์ด๋ฏธ์ง resize ๋ฐ ์ ์ฅ
      for image_path in glob(root_path +'/'+ species_nm + '/' + sub_path + '/*'): 
        image_file_name = image_path.split('/')[-1]
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img.save(root_path +'/'+ species_nm + '_resize/' + sub_path + '/' + image_file_name)
  ```

## 3. ๋ชจ๋ธ๋ง
<p align="center">
<img src = 'https://postfiles.pstatic.net/MjAyMjAzMTVfMjMz/MDAxNjQ3MzE5MzI5MjI4.CAxfbx6lZKslVNr3rxU0zRiijUd_9tDXNvf_nEl7or8g._MuxaQOaULAjEVv8QOkZRfZbaCXyLMzQE92DdZz_NAog.PNG.isanghada03/SE-13d56f79-5ff1-458f-8e2c-8b422cdb29ef.png?type=w966' width="50%" height="50%"></p>  

- ๋ชจ๋ธ ๊ตฌ์กฐ
  - after_net, before_net : CNN์ ํตํด ์ด๋ฏธ์ง์์ ์์ก ๊ธฐ๊ฐ์ ๋์ถํ๋ ๋ชจ๋ธ
    - [Lenet, Mobilenet_v2, Resnext50_32xd4]์ผ๋ก ๊ตฌ์ฑ
  - CompareNet : ๊ฐ ์ด๋ฏธ์ง์์ ๋์ถํ ๊ฒฐ๊ณผ๋ก ๋ ์ด๋ฏธ์ง ๊ฐ์ ์์ก ๊ธฐ๊ฐ์ ๋์ถํ๋ ๋ชจ๋ธ

### Lenet
<p align="center"><img src='https://postfiles.pstatic.net/MjAyMjAzMTVfMTM1/MDAxNjQ3MzE3NzA0ODc2.FMveD6zMA6VkEA-KodxsB6IMcIvYn2DPesq57NIKNqcg.YFUxzbJwiSPww9Gyf6EzP5Xkrc42nyS9HNbIH8iXuQkg.PNG.isanghada03/image.png?type=w966'></p>

- ๊ฐ์ฅ ๊ธฐ๋ณธ์ ์ธ CNN(Convolutional Neural Network)์ผ๋ก Layer ์๋ ฅ๊ฐ๊ณผ ์ถ๋ ฅ๊ฐ์ ์ด๋ฏธ์ง ํฌ๊ธฐ์ ๋ง๊ฒ ์์ 
- ๊ฐ๋จํ ๋ชจ๋ธ๋ก ๋์ ๊ณผ์ ์ ์ดํด๋ณด๊ธฐ ์ํด ์งํ

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
- ์ฌ์  ํ์ต ๋ชจ๋ธ์ ์ฌ์ฉํด ๊ตฌํํ ๋ชจ๋ธ์ ์ฑ๋ฅ์ ๋์ด๊ณ ์ ์๋
- ๊ฐ๋ฒผ์ฐ๋ฉด์ ์ฑ๋ฅ์ด ์ข๋ค๊ณ  ์๋ ค์ง Mobilenet_v2 ์ฌ์ฉ
- EPOCHS ๋ณ๊ฒฝ, Image Augmentation ๋ฑ ๋ค์ํ ๋ฐฉ๋ฒ์ ํตํด ์ฑ๋ฅ ํฅ์ ์๋
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
- Mobilenet_v2๋ณด๋ค ๋ณต์กํ๊ณ  ์ฑ๋ฅ์ด ์ข์ ๋ชจ๋ธ๋ก ๋ณ๊ฒฝํ์ฌ ๋น๊ต ์งํ
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

## 4. ์ต์ข ๊ฒฐ๊ณผ
- ๋ฐ์ด์ฝ ์ ์ถ ๊ฒฐ๊ณผ
  - public : ์ ์ฒด ํ์คํธ ๋ฐ์ดํฐ ์ค 25%
  - private : ์ ์ฒด ํ์คํธ ๋ฐ์ดํฐ ์ค 75%

<p align='center'><img src='https://postfiles.pstatic.net/MjAyMjAzMTVfMjQg/MDAxNjQ3MzIyMjk0OTI1.bWLSCrZi-tHPAOomuuvA1MsO4vlG3x6bWdG-IOzm-L0g.WWasJJvFlqFtzPA4ZQg_Sh1GoH5HHxdPzCTf9FAM56Ig.PNG.isanghada03/%EA%B7%B8%EB%A6%BC2.png?type=w966' width="90%" height="90%"></p>

- Feature Map
  - CNN์ด ์ด๋ฏธ์ง๋ฅผ ๋ถ์ํ๋ ๋ฐฉ์์ ์ดํด๋ณด๊ณ ์ ์งํ
  - Lenet๊ณผ Mobilenet_v2๋ก ์งํ
  - [Lenet]  
  <img src='https://postfiles.pstatic.net/MjAyMjAzMTVfODUg/MDAxNjQ3MzIzMzM2Nzkw.cYt21LFNY90aL2X_3IOKpnDAygkuPJdRm-R3hxo21Cgg.Cq25sAMBdFiaiEXJjxtNwsC92WCXsySdQzyFcRp_E4Qg.PNG.isanghada03/%EA%B7%B8%EB%A6%BC3.png?type=w966' width='70%' height='70%'>
  
  - [Mobilenet_v2]  
  <img src='https://postfiles.pstatic.net/MjAyMjAzMTVfMjEy/MDAxNjQ3MzIzMzM2NjMy.UxrVQfGGRdTg7H4hGVSVBEIcN7M5DsuVbTtzn6cQcxYg.BoUPIuaN7mXmfaKXi-71xU3qZN9R2ZoTTpkXcgCtRbQg.PNG.isanghada03/%EA%B7%B8%EB%A6%BC4.png?type=w966' width='70%' height='70%'>  
  
  - **์ฃผ๋ก ์์ด ๊ฐ์กฐ๋๋ ๊ฒ ํ์ธ**

## 5. ํ๊ณ ๋ฐ ๋ณด์์ 
#### ๐ ์์คํ ์ฌ์ ๋ถ์กฑ
- ์ฌ์ฉ ๋ฐ์ดํฐ๊ฐ ์ด๋ฏธ์ง์ด๊ณ  ์ฌ์  ํ์ต ๋ชจ๋ธ์ Layer๊ฐ ๋์ ์์คํ ์ฌ์์ด ํ์ํ๋ค.
- Colab์ ํตํด ์งํํ  ์๋ ์์์ง๋ง, ์ฃผ๋ก GPU ๋ฉ๋ชจ๋ฆฌ ์ด์๊ฐ ๋ฐ์ํ์ฌ ์งํ์ด ๋งํ ๊ฒฝ์ฐ๊ฐ ์์๋ค.
  - ์ฌ์ ๋ถ์กฑ๊ณผ ํ์ต ์๊ฐ์ด ์ค๋ ๊ฑธ๋ ค ์ฃผ๋ก epoch์ ๋ณ๊ฒฝํ๋ฉฐ ๋น๊ตํ์๋ค.
- ์ฌ์์ด๋ ํ์ต ์๊ฐ ๋ฌธ์ ๊ฐ ํด๊ฒฐ์ด ๋๋ค๋ฉด ์ต์ ์ ์กฐ๊ฑด์ ํ์ํ๊ณ  ์ฌ๋ฌ ๋ชจ๋ธ๋ค์ ์์๋ธํ๋ฉด ์กฐ๊ธ ๋ ์์ฑ๋ ๋์ ๋ชจ๋ธ์ด ๋  ์ ์์ ๊ฒ์ด๋ผ ์๊ฐํ๋ค.

## 6. ์ฐธ๊ณ  ์๋ฃ
- ๋ฐ์ด์ฝ ์ฝ๋ : https://dacon.io/competitions/official/235851/codeshare
- mobilenet_v2 : https://pytorch.org/vision/main/generated/torchvision.models.mobilenet_v2.html
- resnext50_32x4d : https://pytorch.org/vision/main/generated/torchvision.models.resnext50_32x4d.html

## 7. ๊ตฌ์ฑ ์ธ์
- ๊น๋จ๊ท (<a href = 'https://github.com/Isanghada' href='_blank'>github</a>)
- ๋ฐ์ด์  (<a href = 'https://github.com/happyfranc' href='_blank'>github</a>)
- ์ ํ์ค (<a href = 'https://github.com/hyunjuyo' href='_blank'>github</a>)
