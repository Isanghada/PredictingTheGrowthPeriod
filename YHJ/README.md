# [ DL_Project1 ] 생육 기간 예측 프로젝트

## 목적 및 진행 방향<br/>
&nbsp;&nbsp;\- **Part 1)** 한 쌍의 작물 이미지를 입력받아 해당 작물의 생육 기간을 예측하는 모델 개발 및 성능 테스트<br/>
&nbsp;&nbsp;\- **Part 2)** 학습된 모델 기반 Feature Map 시각화를 통해 Convolutional Layer가 이미지를 이해하는 방식에 대해 살펴보기

## 데이터 정보 및 학습 진행 방식
* DACON의 "생육 기간 예측 경진대회"에서 제공된 데이터 활용
* 2개 작물(청경채, 적상추)에 대한 생육 기간 경과일자별 이미지 데이터 저장<br/>
\- 총 753개(청경채 353개, 적상추 400개)
* 작물별 이미지 2장씩을 다양하게 조합하여 2장의 이미지간 경과일 기준 학습 및 모델 성능 테스트 진행
* 모델 평가 기준 : RMSE(Root Mean Squared Error)
<br/>

# Part 1) MobileNet 기반 모델 성능 향상 테스트

## 주요 진행 경과

### 1) Baseline구성 및 데이터셋 생성 --> "DL_project1_v0.3c_유현준.ipynb"

* baseline 구성 및 기본 셋팅<br/>
\- 모델 : mobilenet_v2 활용<br/>
\- learning rate : 0.00005
* 데이터 전처리<br/>
\- 이미지 사이즈가 크므로 효과적인 학습 진행을 위해 이미지 Resize
* 데이터셋 생성<br/>
\- 작물별 2개 이미지 랜덤 추출 및 각각의 time_delta 계산

-----
### 2) 기본 셋팅 및 테스트(1차) --> "DL_project1_v0.5c_유현준.ipynb"

* 주요 셋팅 현황<br/>
\- 모델 : mobilenet_v2 활용<br/>
\- learning rate : 0.00005<br/>
\- epochs : 10<br/>
\- batch_size : 64

* 데이터셋 생성 현황<br/>
\- 작물별 2개 이미지 랜덤 추출한 5000개 데이터 기준 학습 진행<br/>
\- train, validation 비율 => 9 : 1

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 8.109<br/>
\- private : 8.243

-----
### 3) 추가 셋팅 및 테스트(2차) --> "DL_project1_v0.7c_유현준.ipynb"

#### 추가 셋팅 1. Normalize transformation(현 데이터 기준 산출한 평균, 표준편차 사용)

* 내용<br/>
\- 현재 이미지 데이터 기준, RGB 평균 및 표준편차 산출 후 적용

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 8.053<br/>
\- private : 8.278

#### 추가 셋팅 2. 상하반전, 좌우반전 transformation

* 내용<br/>
\- RandomHorizontalFlip(p=0.5) 적용,<br/>
\- RandomVerticalFlip(p=0.5) 적용

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 6.875<br/>
\- private : 7.010

-----
### 4) 추가 셋팅 및 테스트(3차) --> "DL_project1_v0.92c_유현준.ipynb"

#### 추가 셋팅 1. 이상치 검토 및 훈련 데이터 제외

* 내용<br/>
\- 훈련 데이터 중 "BC_03" 폴더 이미지의 경우 작물의 생육 상태가 이상치로 판단되어 제외 처리 후 테스트<br/>
\- 테스트 결과 오히려 제외 전보다 모델의 성능이 떨어진 것으로 확인됨 => 제외 전 데이터 기준으로 이후 테스트 진행

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 7.078<br/>
\- private : 7.317

#### 추가 셋팅 2. 90도 회전 transformation

* 내용<br/>
\- RandomRotation(90) 적용<br/>
\- 로테이션만 추가 반영한 것임에도 앞선 테스트(2차) 대비 모델 성능이 향상된 것으로 확인됨

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 6.326<br/>
\- private : 6.419

#### 추가 셋팅 3. Normalize transformation(Imagenet 데이터셋 기준 산출된 평균, 표준편차 사용)

* 내용<br/>
\- Imagenet 데이터셋 기준으로 산출된 평균 및 표준편차를 사용하여 테스트 진행<br/>
\- 테스트 결과 public 기준으로는 소폭 성능이 향상되었으나, private 기준으로는 약간 성능이 저하된 것으로 확인됨<br/>
\- 종합적으로 봤을 때, 기존 대비 유의미한 성능 향상은 없는 것으로 판단됨 => 현재 데이터 기준으로 산출한 평균, 표준편차를 사용하여 이후 테스트 진행

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 6.278<br/>
\- private : 6.502

-----
### 5) 현 모델 기준 epochs 셋팅 및 테스트(4차) --> "DL_project1_v1.1c_유현준.ipynb"

#### 1. epochs 15

* 내용<br/>
\- 오히려 epochs 10일 때 보다 모델의 성능이 떨어진 것으로 확인됨

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 6.498<br/>
\- private : 6.566

#### 2. epochs 20

* 내용<br/>
\- epochs 15는 물론이고, epochs 10일 때 보다 모델의 성능이 향상된 것으로 확인됨

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 6.067<br/>
\- private : 6.347

#### 3. epochs 19

* 내용<br/>
\- 20번의 epoch 중, validation 데이터 기준 성능이 가장 좋게 나타났던 model state를 불러와 테스트 진행<br/>
\- 지금까지의 테스트 중, 가장 좋은 성능을 나타냄 => public 기준 RMSE 5점대 진입

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 5.820<br/>
\- private : 6.040

-----
### 6) learning rate 조정, epochs 셋팅 및 테스트(5차) --> "DL_project1_v1.3c_유현준.ipynb"

#### 1. lr 0.00003, epochs 30

* 내용<br/>
\- 기존 20번의 epoch 결과 model state를 불러온 뒤, 조정한 learning rate 기준으로 추가 10 epoch 진행함<br/>
\- 기대와 달리 오히려 기존 대비 성능이 저하된 것으로 확인됨

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 6.323<br/>
\- private : 6.510

#### 2. lr 0.00003, epochs 28

* 내용<br/>
\- 30번의 epoch 중, validation 데이터 기준 성능이 가장 좋게 나타났던 model state를 불러와 테스트 진행<br/>
\- public 기준 다시 5점대에 진입하긴 했으나, 기존 모델의 성능 대비 향상되지는 않은 것으로 확인됨

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 5.873<br/>
\- private : 6.080

#### 3. lr 0.00001, epochs 40

* 내용<br/>
\- 기존 30번의 epoch 결과 model state를 불러온 뒤, 조정한 learning rate 기준으로 추가 10 epoch 진행함<br/>
\- 기대와는 달리 성능이 오히려 저하된 것으로 나타남

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 6.407<br/>
\- private : 6.640

#### 4. lr 0.00001, epochs 32

* 내용<br/>
\- 40번의 epoch 중, validation 데이터 기준 성능이 가장 좋게 나타났던 model state를 불러와 테스트 진행<br/>
\- 마찬가지로 기존 대비 성능이 좋지 못한 것으로 확인됨 => 본 모델 기준으로 약 20 epochs 이후에는 과적합이 나타난 것으로 추정됨

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 6.247<br/>
\- private : 6.454

-----
### 7) 2개 모델 예측결과 앙상블 및 테스트(6차)

#### 1. 성능이 가장 좋았던 2개 모델 예측결과 앙상블

* 내용<br/>
\- epochs 19 모델과 epochs 28 모델이 예측한 결과를 5:5로 앙상블 진행<br/>
\- 테스트 결과, 큰 폭은 아니지만 아래와 같이 SCORE가 약간 개선되었음<br/>
\- 이는 지금까지의 테스트 중 가장 좋은 SCORE로, 서로 다른 모델간의 앙상블로 인해 성능이 개선된 것으로 확인됨

* 테스트 결과 SCORE (DACON 제출 후 산출된 점수 기준)<br/>
\- public : 5.812<br/>
\- private : 6.026
<br/>

# Part 2) Feature Map 시각화

### 1) LeNet 기반 모델 학습 진행 --> "DL_project1_featuremap_v1.1c__유현준.ipynb"

* 기본적인 LeNet 구조를 활용해 CNN이 이미지를 이해하는 방식에 대해 살펴보고자 함<br/>
\- 아래와 같은 구조의 첫번째 레이어에서 Feature Map을 뽑아낸 뒤, 추출된 8개 채널별로 시각화 및 비교 예정
![image](https://user-images.githubusercontent.com/76440511/147557433-7c273e9b-989d-4299-8496-414ee0183f09.png)

* 단계별 학습 진행<br/>
\- 학습 단계는 앞서 MobileNet에 적용했던 기준과 동일하게 적용함<br/>
&nbsp;&nbsp;&nbsp;&nbsp;\- lr 0.00005 기준으로 20 epochs 진행<br/>
&nbsp;&nbsp;&nbsp;&nbsp;\- lr 0.00003 기준으로 추가 10 epochs 진행 => 누적 30 epochs<br/>
&nbsp;&nbsp;&nbsp;&nbsp;\- lr 0.00001 기준으로 추가 10 epochs 진행 => 누적 40 epochs<br/>
\- 학습된 모델의 상태 및 성능수준에 따라 동일한 이미지에 대해 Feature Map이 어떻게 달라지는지 비교해보고자 함

* 모델의 최종 성능이 중요한 Part는 아니므로, 절대적인 모델의 성능 수치보다는 학습 초반과 후반의 상대적인 성능 변화 수치를 참조하며 진행하고자 함

-----
### 2) 학습된 LeNet 모델별 Feature Map 시각화 및 비교 --> "DL_project1_featuremap_v2c__유현준.ipynb"

* 첫번째 epoch 시점의 모델 기준 Feature Map 확인<br/>
\- 해당 모델의 VALIDATION_LOSS MAE : 5.205 수준
![image](https://user-images.githubusercontent.com/76440511/147561302-d33776c2-7238-47e3-a0c3-d4fe1093e8c6.png)

* 10번째 epoch 시점의 모델 기준 Feature Map 확인<br/>
\- 해당 모델의 VALIDATION_LOSS MAE : 3.469 수준
![image](https://user-images.githubusercontent.com/76440511/147570265-79ca2caa-abf9-4841-b255-1dd1efbecad4.png)

* 39번째 epoch 시점의 모델 기준 Feature Map 확인<br/>
\- 해당 모델의 VALIDATION_LOSS MAE : 2.290 수준
![image](https://user-images.githubusercontent.com/76440511/147570382-8f4e24c3-a217-432c-bca0-bdd4ee3d0282.png)

* Feature Map 1개 채널 확대하여 모델간 비교

|첫번째 epoch 시점의 모델 기준|39번째 epoch 시점의 모델 기준|
|-|-|
|![image](https://user-images.githubusercontent.com/76440511/147570819-7bc4e5da-e807-4247-a4d2-f3a1643baeac.png)|![image](https://user-images.githubusercontent.com/76440511/147570853-9982940b-e2f9-4e6b-acf8-e1a51fd85a05.png)|

