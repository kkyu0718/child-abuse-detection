# AI CCTV System Child Abuse Detection

- **팀원** : 데이터 청년 캠퍼스 10조 김규민, 김규원, 김석기, 김호준, 이은서

- **프로젝트 개요** : CNN + BiLSTM을 통한 CCTV 실시간 아동학대 감지 시스템

- **프로젝트 작업 기간** : 2021.07.04 ~ 2021.08.28

## 💡 Service Flow

![그림1](https://user-images.githubusercontent.com/80209277/134019304-5853260e-75ce-4565-8ad9-a56c97cafc6b.png)

## 📁 Data information

- 성인 폭력 데이터 for model training
- 아동 폭력 데이터 for transfer learning

## 📌 Model Input & 전처리

![그림8](https://user-images.githubusercontent.com/80209277/134020030-e39e3b87-1b17-4cff-a06f-f31b2420326d.png)

- **Input** : 폭력 데이터 video
- **Key frame sampling** : video 당 5개의 프레임 추출
- **Frame resize** : vgg input 위한 frame 크기 조정
- **Frame difference** : frame 간의 차이 연산
- **Normalization** : 연산 편이 위한 조정


## 📈 Model 

**성인 폭력 데이터 학습** : CNN (VGG19) + BiLstm

<img src="https://user-images.githubusercontent.com/80209277/134022985-23591d8c-2254-4e45-814d-70adf920c72e.png" width="600" height="200"/>

**아동 폭력 데이터 전이학습** : Transfer learning

<img src="https://user-images.githubusercontent.com/80209277/134023002-385ee90a-4e1b-4e1d-8915-da4690bbb899.png" width="600" height="200"/>
