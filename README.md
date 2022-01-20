# 파일 설명

전처리 및 모델 코드

- preprocess.py : 전처리 과정 함수를 포함한다. modeling.ipynb에서 import 하여 사용한다.
- modeling.ipynb : 모델 구축 및 최적화 과정



빅콘테스트 홈페이지에서 다운로드 한 문제 엑셀 파일은 이름만 바꿔 그대로 사용했다.

- flood_data.xlsx: 제공 데이터
- eval_data.xlsx: 평가 데이터



외부 데이터 수집을 위한 코드는 crawling 폴더에 모아두었다.

- crawling/crawling.ipynb : 웹사이트에 접근하여 외부 데이터를 수집한다. 수집된 데이터는 other_dams.xlsx에 저장한다.
- crawling/process_other_dams.ipynb : 수집한 데이터인 other_dams.xlsx를 읽고 pandas DataFrame으로 변환하여 python pickle 파일인 other_dams.pkl에 저장한다. modeling.ipynb에서는 이 other_dams.pkl을 열어서 외부 데이터를 활용한다.
