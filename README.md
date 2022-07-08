# 2021-Big-Contest
댐 유입 수량 예측을 통한 최적의 수량 예측 모형 도출

## Background
[2021 빅콘테스트 대회소개_홍수ZERO부분](https://github.com/ChoiDae1/2021-Big-Contest/blob/main/05_%EC%A0%9C9%ED%9A%8C%202021%20%EB%B9%85%EC%BD%98%ED%85%8C%EC%8A%A4%ED%8A%B8%20%EB%AC%B8%EC%A0%9C%EC%84%A4%EB%AA%85_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D%EB%B6%84%EC%95%BC_%ED%93%A8%EC%B2%98%EC%8A%A4%EB%A6%AC%EA%B7%B8_%ED%99%8D%EC%88%98ZERO_210730.pdf)에 참가하여 진행한 프로젝트이다.

![image](https://user-images.githubusercontent.com/95220313/150293922-0b6cc15b-f135-49f7-9653-6952277a7916.png)

## Report
[2021 빅콘테스트 대회소개_홍수ZERO부분_최종보고서](https://github.com/ChoiDae1/2021-Big-Contest/blob/main/ppt(%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C).pdf)

## Description

### Datasets

> 저작권 문제로 인해 데이터셋은 레포지토리에 포함되지 않는다.

빅콘테스트 홈페이지에서 다운로드 한 문제 엑셀 파일은 이름만 바꿔 그대로 사용했다.

- `flood_data.xlsx`: 제공 데이터
- `eval_data.xlsx`: 평가 데이터

### Pre-processing and ML Models

- `preprocess.py`: 전처리 과정 함수를 포함한다. `modeling.ipynb`에서 import 하여 사용한다.
- `modeling.ipynb`: 머신러닝 모델 구축 및 최적화 과정

### Crawling

외부 데이터 수집을 위한 코드는 `crawling` 디렉토리에 모아두었다.

- `crawling.ipynb`: 웹사이트에 접근하여 외부 데이터를 수집한다. 수집된 데이터는 `other_dams.xlsx`에 저장한다.
- `process_other_dams.ipynb`: 수집한 데이터인 `other_dams.xlsx`을 읽고 pandas DataFrame으로 변환하여 python pickle 파일인 `other_dams.pkl`에 저장한다. `modeling.ipynb`에서는 이 `other_dams.pkl`을 열어서 외부 데이터를 활용한다.

## Technology Stack

* Python
* Numpy, pandas - 데이터 전처리
* Selenium - 외부 웹사이트 데이터 크롤링
* scikit-learn, XGBoost - 머신러닝 모델 구축

## Contributors

* [최대원](https://github.com/ChoiDae1): 머신러닝 모델링
* [임정섭](https://github.com/jseop-lim): 데이터 가공, 크롤링
