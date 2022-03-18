# 다양한 로지스틱 회귀

import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head() # 처음 5개의 행을 출력

- unique() : 원하는 열에서 고유한 값을 추출
- 위 데이터프레임에서는 species열은 타깃으로 하고 남은 열을 입력데이터로 사용


print(pd.unique(fish['Species']))

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height','Width']].to_numpy()

print(fish_input[:5])

fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)  # 훈련세트의 통계값으로 테스트세트를 변환하는것 잊지 않기!!!
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

### K - 최근접 이웃 분류기의 확률 예측


from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors = 3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

- 타깃데이터에 2개 이상의 클래스가 있으면 ***다중분류*** 라고 한다.
- 앞에서 타깃데이터를 (0,1)로 나눈것은 이진분류다. 
- 다중분류에서도 타깃값을 숫자로 표현 가능하지만 사이킷런에서는 문자열로 된 타깃값 사용 가능
- 주의할 점) 타깃값을 사이킷런 모델에 전달하면 순서가 자동으로 알파벳순서로 바뀜
    - 타깃데이터들은 classes_ 속성에 들어가 있음

print(kn.classes_)

print(kn.predict(test_scaled[:5]))

- predict_proba() 에서드의 출력순서는 classes_속성과 같다. 
- 즉, 첫번째 열이 Bream의 확률, 두번째 열이 Parkki의 확률이다.

import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals =4))

distance, index = kn.kneighbors(test_scaled[3:4])
print(train_target[index])

### 로지스틱 회귀
- 이름은 회귀이지만 분류모델
- 로지스틱 회귀는 선형회귀와 동일하게 선형방정식을 학습
- z = a X (Weight) + b X (Length) + c X (Diagonal) + d X (Height) + e X (Width) + f
- 여기에서 a,b,c,d,e는 가중치 혹은 계수이다. 
- z는 어떤 값도 가능하지만 확률이 되려면 0~1 사이 값이 되야 한다. 
    - 아주 큰 음수일 때는 0이 되고 아주 큰 양수일 때는 1이 되도록 바꾼 함수 - 시그모이드 함수
        - 시그모이드 함수 = <img src="https://cdn-images-1.medium.com/max/1200/1*Vo7UFksa_8Ne5HcfEzHNWQ.png" width="200" height="80">
        - 지수함수 계산은 np.exp()함수 사용



import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1/(1 + np.exp(-z))
plt.plot(z,phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

### 로지스틱 회귀로 이진 분류 수행하기

- 넘파이 배열은 True, False값을 전달하여 행을 선택 가능
- 이를 ***불리언 인덱싱*** 이라고 함

char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])

- 도미(bream)와 빙어(Smelt)의 행만 골라내기

bream_smelt_index = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_index]
target_bream_smelt = train_target[bream_smelt_index]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))

- predict_proba() : 예측확률을 출력

print(lr.predict_proba(train_bream_smelt[:5]))

- Bream과 Smelt중 누가 양성 클래스인가
- 타깃값이 알파벳순이기 때문에 빙어(Smelt)가 양성 클래스이다.

print(lr.classes_)

- 가중치와 절편 확인하기

print(lr.coef_, lr.intercept_)

 - train_bream_smelt의 처음 5개 샘플의 z값을 확인

decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

- 이 z값들을 시그모이드 함수에 통과시키면 확룰을 얻을 수 있다.
- 파이썬의 사이파이(scipy)라이브러리에도 시그모이드 함수 expit() 가 있다.  

from scipy.special import expit
print(expit(decisions))

### 로지스틱 회귀로 다중분류 수행

- LogisticRegression 클래스는 기본적으로 반복적인 알조리즘을 사용
    - max_iter 매개변수에서 반복횟수를 지정하며 기본값은 100
- LogisticRegression 클래스는 기본적으로 릿지회귀와 같이 계수의 제곱을 이용해 규제함
- 릿지회귀는 alpha로 로지스틱회귀는 C 이다
- 매개변수 C의 기본값은 1 이고 alpha와 달리 값이 작을수록 규제가 커

lr = LogisticRegression(C=20, max_iter = 1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.predict(test_scaled[:5]))

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals = 3))

- 타깃 클래스 정보 확인

print(lr.classes_)

- coef_ 와 intercept_ 의 행의 개수가 7개 = 다중분류는 클래스마다 z값을 하나씩 계산 => 가장 높은 z값의 클래스가 예측 클래스

- 확률을 계산하기
    - 이진분류에서는 시그모이드 함수를 사용
    - 다중분류는 소프트맥스(softmax)함수를 사용
- 7개의 z값을 z1 ~ z7로 가정
e_sum = e<sup>z1</sup> + e<sup>z2</sup> + e<sup>z3</sup> + e<sup>z4</sup> + e<sup>z5</sup> + e<sup>z6</sup> + e<sup>z7</sup>
- 그 다음 e<sup>z1</sup> ~ e<sup>z7</sup> 까지 e_sum으로 나누면 됨
- 그럼 각각의 값이 확률


print(lr.coef_.shape, lr.intercept_.shape)

decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals = 2))

from scipy.special import softmax
proba = softmax(decision, axis =1)
print(np.round(proba, decimals = 3))

# 확률적 경사 하강법

 점진적 학습 : 전에 훈련된 모델을 버리지 않고, 앞으로 추가될 데이터들을 조금식 더 훈련하는 방법
- 점진적 학습의 대표적인 알고리즘이 ***확률적 경사 하강법***이다
    - 경사 하강이란 이미지는  산길을 내려갈 때 가장 빠른 길은 경사가 가장 가파른 길이다
        - 확률적 경사 하강법에서 훈련세트를 한번 모두 사용하는 과정을 ***에포크***라고 한다.
            - 일반적으로 경사하강법은 수십, 수백번 수행한다. 
        - 확률이란 랜덤하게 train_input데이터에서 데이터를 하나씩 뽑는 것이다. 
            - 하나의 데이터씩 말고 무작위의 몇개의 샘플을 사용해서 경사를 따라내려가는 경사하강법을 ***미니배치 경사 하강법*** 이라고 한다
            - 극단적으로 한번 경사로를 따라 내려가기 위해 전체 샘플을 사용하는 방법을 ***배치경사 하강법***이라고 한다.
    - 여기서 내려오는 산은 ***손실함수***라 부르는 산이다
        - 손실함수는 어떤 문제에서 머신러닝 알고리즘이 얼마나 엉터리인지 측정하는 기준
        - 손실함수의 값이 작을수록 좋지만, 어떤 값이 최솟값인지 모른다.
            -  그래서 가능한 많이 찾아보고 만족할만한 수준이면 산을 내려왔다고 인정해야 함
        - 산의 경사면은 연속적이므로 손실함수도 연속적, 즉 미분가능해야 한다

