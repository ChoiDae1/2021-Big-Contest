import numpy as np
import pandas as pd


#### 엑셀 파일을 DataFrame으로 변환

def create_dataframe(filename, eval=False, flood_id=False, period_col=False):
    '''엑셀 파일 읽고 적절한 인덱스가 포함된 DataFrame으로 변환하여 피쳐와 라벨을 튜플로 반환한다.
    평가데이터(홍수사상 26)가 제거된다.'''
    
    # 엑셀 파일 불러오기
    with pd.ExcelFile(filename) as xlsx:
        raw_index_df = pd.read_excel(xlsx, skiprows=2, header=None, usecols=range(0, 5))
        raw_index_df.columns = ['flood_id', 'year', 'month', 'day', 'hour']
        
        raw_period = pd.PeriodIndex(pd.to_datetime(raw_index_df.iloc[:, 1:]), freq='H', name='period')
        raw_index = pd.MultiIndex.from_arrays([raw_index_df['flood_id'], raw_period])
        
        raw_data = pd.read_excel(xlsx, header=[0, 1]).iloc[:, 5:]
        raw_data.index = raw_index
    
    
    X_df = raw_data.iloc[:, 1:]
    X_df.columns = _create_X_columns()
    
    if flood_id:
        X_df['flood_id'] = pd.Series(raw_index_df['flood_id'].values, index=raw_index)
    
    if period_col:
        raw_period_df = raw_index_df.drop('flood_id', axis=1)
        X_df[raw_period_df.columns] = raw_period_df.set_index(raw_index)      
    
    
    y_df = raw_data.iloc[:, 0].rename('inflow')
    
    if eval == False:
        X_df = _drop_evaluation(X_df)
        y_df = _drop_evaluation(y_df)
    
    return X_df, y_df


# (내부함수)
def _create_X_columns():
    '''X의 columns 인덱스를 생성'''
    numbers = np.arange(1, 7)
    measure = ['rainfall_mean_all', 'rainfall_A', 'rainfall_B', 'rainfall_C', 'rainfall_D', 'waterlevel_E', 'waterlevel_D']
    return pd.MultiIndex.from_product([numbers, measure], names=['dataset', 'variable'])


def _drop_evaluation(df):
    '''data_x_idx, data_y_idx에서 평가데이터 부분(홍수사상 26) 제거'''
    return df.drop(index=26)


# -------------------------------------------------------------------------------------------
#### 정규화 및 표준화

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale(df1, df2=None, df=False, method=''):
    '''df1 기준으로 df2를 정규화/표준화한다. df2가 없으면 df1을 정규화/표준화한다.
    df 인자는 반환형을 결정한다. df=True이면 데이터프레임을, df=False이면 ndarray를 반환한다.'''
    
    if method in ('normalize', 'n'):
        scaler = MinMaxScaler()
    elif method in ('standardize', 's'):
        scaler = StandardScaler()
    else:
        raise ValueError('method must be "n(ormalize)" or "s(tandardize)"')
    
    scaler.fit(df1)
    
    if df2 is None:
        df2 = df1
    
    df_scaled = scaler.transform(df2)
    if df: df_scaled = pd.DataFrame(df_scaled, index=df2.index, columns=df2.columns)
    
    return df_scaled


def take_log(df, columns):
    '''특정 칼럼에 1을 더하고 자연로그를 취한 새로운 DataFrame을 반환한다.'''
    
    df = df.copy()
    
    df[columns] = np.log1p(df[columns]) # 모든 원소는 0이상이며, 0인 것 또한 존재하므로 1을 더하고 자연로그를 취한다.
    
    return df


# -------------------------------------------------------------------------------------------
#### 파생변수 생성
 
def mutate_period_rate(df):
    '''홍수사상별 시간 진행도 칼럼 추가'''
    
    period_rate = df.reset_index('period') \
                    .groupby(df.index.get_level_values(0))['period'] \
                    .transform(lambda x: (x - x.min())/(x.max() - x.min())) \
                    .reset_index(drop=True)
    
    period_rate.index = df.index
    
    return df.assign(period_rate = period_rate)


def mutate_flood_id(df):
    '''홍수사상번호 칼럼 추가'''
    
    return df.assign(flood_id = df.index.get_level_values('flood_id'))


def mutate_dummies(df, columns): # 최종모델에서 미사용
    '''특정 변수에 대한 원-핫 인코딩 칼럼들(dummies) 추가'''
    dummies = pd.get_dummies(df[columns], prefix=columns)
    
    df = df.copy()
    df[dummies.columns] = dummies.set_index(df.index)

    return df


# def create_lags(df, n):
#     '''df(종속변수)에 대한 n개의 지연값 변수를 생성하여 반환'''
#     df_list = []
#     grouped = df.groupby('flood_id')
    
#     for i in range(1, n+1):
#         df_list.append(grouped.transform('shift', i).rename(f'inflow_lag{i}'))
    
#     return pd.concat(df_list, axis=1)


# def create_ma(df, window):
#     '''df(종속변수)의 이동평균 칼럼을 생성하여 반환. window는 창크기'''
#     return df.reset_index('flood_id').groupby('flood_id')['inflow'].rolling(window).mean()


# -------------------------------------------------------------------------------------------
#### 이상치 제거

def drop_outlier(X_df, y_df, columns, method:str, weight=1.5, threshold=2):
    '''이상치가 포함된 행을 제거한다.
    columns는 칼럼 라벨 혹은 라벨 문자열의 리스트이다.'''

    if method == 'iqr':
        is_outlier = get_outlier_iqr(X_df, columns, weight)
    elif method == 'zscore':
        is_outlier = get_outlier_zscore(X_df, columns, threshold)
    else:
        raise ValueError('method must be "iqr" or "zscore"')
    
    if is_outlier.ndim == 2:
        is_outlier = is_outlier.any(axis=1)
    
    return X_df[~is_outlier], y_df[~is_outlier]


def get_outlier_iqr(df, columns, weight=1.5):
    '''사분위수를 이용하여 박스플롯에서 이상치로 분류되는 행의 불리언 배열을 반환한다.'''
    
    quartile_1 = df[columns].quantile(0.25)
    quartile_3 = df[columns].quantile(0.75)
    iqr = quartile_3 - quartile_1
    
    is_outlier = (df[columns] < (quartile_1 - weight*iqr)) | (df[columns] > (quartile_3 + weight*iqr))
        
    return is_outlier.values


def get_outlier_zscore(df, columns, threshold=2):
    '''표준화의 결과인 Z-score의 절댓값이 threshold를 초과하는 행의 불리언 배열을 반환한다.'''

    # scaler = StandardScaler()
    # df_scaled = scaler.fit_transform(df[columns])
    
    # df_scaled = df[columns].transform(lambda x: (x - x.mean())/x.std())
    # df_scaled = (df[columns] - np.mean(df[columns], axis=0)) / np.std(df[columns], axis=0)
    
    df_scaled = (df[columns] - df[columns].mean()) / df[columns].std() # sklearn 이외에 가장 빠른 방법 선택

    is_outlier = (np.abs(df_scaled) > threshold) # (Series)/DataFrame
    
    return is_outlier.values


def replace_outlier(df, columns, method:str, weight=1.5, threshold=2):
    ''''이상치를 각 변수의 대표값으로 변경한다.
    method='iqr': 사분위수를 이용하여 박스플롯에서 이상치로 분류되는 행의 값을 중앙값으로 대체한다.
    method='zscore': 표준화의 결과인 Z-score의 절댓값이 threshold를 초과하는 행의 값을 평균값으로 대체한다.
    columns는 칼럼 라벨 혹은 라벨 문자열의 리스트이다.'''
    
    df = df.copy()

    if method == 'iqr':
        is_outlier = get_outlier_iqr(df, columns, weight)
        other = df[columns].median()
    elif method == 'zscore':
        is_outlier = get_outlier_zscore(df, columns, threshold)
        other = df[columns].mean()
    else:
        raise ValueError('method must be "iqr" or "zscore"')
    
    if is_outlier.ndim == 2:
        df[columns] = df[columns].mask(is_outlier, other, axis=1)
    else:
        df[columns] = df[columns].mask(is_outlier, other)
    
    return df


        
if __name__ == '__main__':
    # 엑셀 파일 읽고 적절한 인덱스가 포함된 DataFrame으로 변환하여 feature 데이터프레임과 label 시리즈를 튜플로 반환한다.
    # 평가데이터(홍수사상 26)가 제거된다.
    X_flood_df, y_flood_df = create_dataframe('flood.xlsx', period_col=True)
    
    # mutate_로 시작하는 함수는 매개변수로 인덱스가 포함된 feature 데이터프레임을 받는다.
    # 인자로 전될단 데이터프레임의 복사본을 반환하므로 X를 갱신하려면 아래 형태로 쓴다.
    # 각 함수의 구체적 기능은 함수 정의를 참고하라.
    X_flood_df = mutate_period_rate(X_flood_df)
    X_flood_df = mutate_flood_id(X_flood_df)
    
    # 날짜와 홍수사상번호로 구성된 인덱스를 제거하려면 아래와 같이 쓴다.
    X_flood_df = X_flood_df.reset_index(drop=True)

    # 정규화는 아래와 같다.
    # df 인자는 반환형을 결정한다. df=True이면 데이터프레임을, df=False이면 ndarray를 반환한다.
    print(scale(X_flood_df, method='n'))
    print(scale(X_flood_df, df=True, method='s'))

    print(y_flood_df)
       
    # 이상치 제거 확인하기
    X_droped_iqr, y_droped_iqr = drop_outlier(X_flood_df, y_flood_df, [(1, 'waterlevel_E'), (2, 'rainfall_A')], 'iqr')
    
    print('이상치 제거 전 모양:', X_flood_df.shape)
    print('이상치 제거 후 모양:', X_droped_iqr.shape)
    print(y_droped_iqr)
    
    # 이상치 대체 확인하기
    # 두 변수 중 하나라도 이상치에 해당한다면 해당 행을 삭제한다(OR 연산).
    X_replaced_zscore = replace_outlier(X_flood_df, [(1, 'waterlevel_E'), (2, 'rainfall_A')], 'zscore')
    print(X_replaced_zscore[(1, 'waterlevel_E')].value_counts())
    
    X_replaced_iqr = replace_outlier(X_flood_df, [(1, 'waterlevel_E'), (2, 'rainfall_A')], 'iqr')
    print(X_replaced_iqr[(1, 'waterlevel_E')].value_counts()) # iqr로 대체하는 경우 quantile의 의미를 살려 중앙값으로 대체했다.
    
    # [참고] 제거와 대체 모두 [(1, 'waterlevel_E'), (2, 'rainfall_A')] 대신 (1, 'waterlevel_E')와 같이 단일 칼럼을 전달해도 동작한다.


    # 두 변수에 로그를 취한다.
    X_flood_df = take_log(X_flood_df, [(1, 'waterlevel_E'), (2, 'rainfall_A')])
    # 한 변수에 로그를 취한다.(위 결과도 중첩)
    X_flood_df = take_log(X_flood_df, (3, 'waterlevel_D'))