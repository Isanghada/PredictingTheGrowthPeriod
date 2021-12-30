# DL_project


생육 기간 예측 프로젝트

* 목적 및 배경 : 한 쌍의 이미지를 입력받아 작물의 생육 기간을 예측하는 모델 개발
* 데이터 : 2개 작물(청경채, 적상추)에 대한 생육 기간 경과일자별 이미지 데이터 저장
   - 학습 : 753개(청경채 353개, 적상추 400개)
   - 테스트 : 307개(청경채 139개, 적상추 168개)

* 분석 방법
   - 작물별 이미지 2장씩을 다양하게 조합하여 2장의 이미지간 경과일을 기준으로 학습 및 평가 진행 예정
   - 모델 평가 기준 : RMSE(Root Mean Squared Error)

* 데이콘 baseline을 기초로 작성
   - 이미지 Resize (224, 224) 진행  
   - 같은 종, 같은 폴더에서 2개씩 1,000개씩 추출하여 8:2로 분리


* Model : LeNet 
   - input : 224 * 224 * 3 Image
   - c1 layer = nn.Conv2d(input_channel_size=3, out_volumn_size=6, kernel_size=5, padding=2), nn.ReLU(),
            224*224 size 이미지를 6개의 5*5 필터와 convolution 연산
   - s2 layer = nn.AvgPool2d(kernel_size=2, stride=2))
            6장 224*224 특성 맵에  2*2를 stride = 2로 설정해서 서브 샘플링 진행, 112*112 사이즈로 특정 맵 축소
   - c3 layer = nn.Conv2d(input_channel_size=6, out_volumn_size=16, kernel_size=5), nn.ReLU(),
            6장의 112*112 사이즈 특성 맵에 convolution 연산 수행해서 16장의 54*54사이즈 특성 맵을 산출 5*5 필터
   - s4 layer = nn.AvgPool2d(kernel_size=2, stride=2))
            16장의 54*54사이즈 특성 맵에  2*2를 stride = 2로 설정해서 서브 샘플링 진행, 46656 축소
   - c5 layer = nn.Flatten(), nn.Linear(16*54*54, 120), nn.ReLU(),  
            평탄화(Flatten)으로 16장의 54*54 사이즈 특성 맵을 FC를 통해 노드를 120으로 축소
   - f6 layer = nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10)
            84개의 유닛을 가진 피드포워드 신경망, c5의 결과를 84개 유닛에 연결
   - output layer = nn.Linear(10, 1)
            최종적으로 이미지가 갖는 결과 산출
            
   - optim : Adam
   - lr = 0.00005
   - epochs = 10
   - batch_size = 64
   - valid_batch_size = 50

* SCORE(같은 폴더 내 추출)
   - public : 13.244482
   - private : 12.55080795288086

* SCORE(같은 종내 무작위 추출) > Data augmentation(증가) 가능성
   - public : 15.292053
   - private : 14.207005500793457
