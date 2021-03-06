# 프로젝트 소스 코드
### 개요
- <a href = 'https://drive.google.com/drive/folders/1UnQLfHjjKR-ePtaHK7WnkqUZ48yb1YBm?usp=sharing' target='_blink'>구글 드라이브</a> : 데이터, 모델, 결과, 코드
- 프로젝트 진행 : Google Colab 사용
- 한 쌍의 이미지를 입력 값으로 받아 작물의 생육 기간을 예측하는 모델 개발
  - 현재는 성장 기간 예측만 진행하지만 회차가 진행되며 환경 변수를 추가로 제공할 예정
- <a href='https://dacon.io/competitions/official/235851/overview/description' target='_blink'>데이콘</a> 데이터 사용
  - 훈련용 이미지 : 총 753개
    - 청경채 : 353개
    - 적상추 : 400개
  - 테스트 이미지 : 총 307개
    - 청경채 : 139개
    - 적상추 : 168개
  - 작물별 이미지 2장씩을 조합하여 2장의 이미지간 경과일 기준으로 학습 및 평가 진행
- 모델 평가 기준 : RMSE(Root Mean Squared Error)
---
- **DL_project_v0.1_김남규.ipynb** : 2021-12-21
  - <a href='https://dacon.io/competitions/official/235851/codeshare/3772?page=1&dtype=recent' target='_blink'>데이콘 baseline</a>을 기초로 작성
  - 기본 이미지를 그대로 사용하면 colab 환경에서 GPU 메모리 초과 발생
  - 이미지 Resize (224, 224) 진행
  - colab GPU 사용 시간 초과로 학습 미진행
  - Model 세팅
    - mobilenet_v2 사용 
    - optim : Adam
    - lr = 1e-5
    - epochs = 10
    - batch_size = 64
    - valid_batch_size = 50
---
- **DL_project_v0.2_김남규.ipynb** : 2021-12-22
  - 코드내 오류 수정
  - Resize 진행한 이미지를 통해 학습 진행
  - Resize 이미지 : <a href='https://drive.google.com/file/d/1YAiw-7hJP9PPy8oIslJuzMq9AMjK81XN/view?usp=sharing' target = '_blink'>구글 드라이브</a>
  - Score
    - public : 11.6930319765
    - private : 10.5438618032
---
- **DL_project_v0.3_김남규.ipynb** : 2021-12-24
  - Train 이미지에 대해 각 디렉토리별 모든 경우의 수를 훈련, 검증 데이터로 활용 : 총 14613개
    - Train Data : bc(6132개), lt(7201개)
    - Valid Data : bc(680개), lt(780개)
  - Train 이미지 Transform
    - Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    - RandomHorizontalFlip(p=0.5)
    - RandomVerticalFlip(p=0.5)
    - RandomAffine((-20, 20))
    - RandomRotation(degrees=(0, 90))
  - Test 추론 결과에 대해 `성장 기간`은 음수가 될 수 없으므로, 음수에 대해 1로 수정
  - Score
    - public : 10.8615013022
    - private : 10.4427667876
---
- **DL_project_v0.3.2_김남규.ipynb** : 2021-12-25(2021-12-29 수정)
  - Model State 저장 시 Optimizer State도 저장
    - 학습을 이어할 수 있도록 하기 위함
  - Train 이미지 미리 불러와 학습 시간 단축
    - Dict형 {'PATH' : IMAGE 파일}
    - Train 이미지 753개 : 청경채-bc(353개), 적상추-lt(400개)
  - Train 이미지 Transform(Vaild Data의 경우 Normalize만 적용)
    - Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    - RandomHorizontalFlip(p=0.5)
    - RandomVerticalFlip(p=0.5)
    - RandomAffine((-20, 20))
    - RandomRotation(degrees=(0, 90))
  - Test 이미지 Transform
    - Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  - Model 세팅(변경점만 작성)
    - EPOCHS = 10, 15
    - BATCH_SIZE = 64 유지 (<= 80으로 변경을 시도해보았으나 Colab GPU 메모리 초과 발생)
  - Score(10)
    - public : 6.5418290462
    - private : 6.7816107639
  - Score(15)
    - public : 6.4125252497
    - private : 6.6128679624
  - 이후 계획
    - EPOCHS의 증가로 Score의 개선이 있는 것으로 보이므로 조금 더 횟수를 늘려 확인
      - 15 EPOCHS까지 학습한 model state, optimizer state가 있으므로 활용해 진행할 예정
    - Feature Map 시각화 시도
    - 사전 학습 모델(현재 mobilenet_v2) 변경
---
- **DL_project_v0.3.3_김남규.ipynb** : 2021-12-26
  - [DL_project_v0.3.2_김남규.ipynb]에서 저장한 Model State 활용
  - Model 세팅(변경점만 작성)
    - EPOCHS = 20
      - 15 EPOCHS까지 진행하였기에 5 EPOCHS만 추가 진행
  - Score
    - public : 6.1323005215
    - private : 6.4400199761
  - 이후 계획
    - EPOCHS 증가로 Score가 좋아지는 것 확인
      - EPOCHS 증가 작업은 최종 모델로 변경한 후 다시 작업 예정
    - Feature Map 시각화 시도
    - 사전 학습 모델(현재 mobilenet_v2) 변경
---
- **DL_project_v0.4_김남규.ipynb** : 2021-12-27
  - resnext50_32x4d 사용
  - 모델 특성 변경
    - EPOCHS = 10
    - BATCH_SIZE = 40 (Colab GPU 메모리 초과로 인해 변경)
    - VALID_BATCH_SIZE = 40
  - Feature Map
    - mobilenet_v2의 결과를 이용하여 Feature Map 시도
    - [1280개의 7 X 7 이미지] 특성 확인 : Sample로 10개만 matplotlib을 통해 시각화
  - Score
    - public : 6.0565558842
    - private : 6.2244916223
  - 이후 계획
    - EPOCHS를 증가시켜 Score 확인
    - Feature Map 시각화 시도
      - 전체 이미지를 확인할 수 있는 정도의 레이어로 시도?
---
- **DL_project_v0.4.1_김남규.ipynb** : 2021-12-28
  - [DL_project_v0.4_김남규.ipynb]에서 저장한 Model State 활용
  - 모델 특성 변경
    - EPOCHS = 20
      - 10 EPOCHS까지 진행하였기에 10 EPOCHS만 추가 진행
  - Feature Map
    - mobilenet_v2(v0.3.3)의 결과를 이용하여 Feature Map
    - 물체(생육의 잎)이 주로 인식(강조)되는 것을 확인 가능
  - Score
    - public : 5.8966784082
    - private : 5.9965117875
  - 계획
    - 코드 및 결과 정리
---
