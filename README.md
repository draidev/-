# 신용카드 사용자 연체 예측(AI SCHOOL 5기 Semi-project3)
기간 : 2022-05-02 ~ 2022-05-10
팀원 : 김나리(팀장), 서민정, 이찬영, 전준용, 정연준

---
## 01. occyp_type열 결측치 채우기 논의
- occyp_type 결측치 → 8171개  
- DAYS_EMPLOYED로 양수값 무직 처리하면 → 3700개 남음  
- income_type열을 기반으로 income_type의 value가 Commercial associate 이면 그 중에서 가장 많은 직업을 결측치에 넣는 형식으로 하기로 함  
- 추후에 성능 개선을 위해 income_total과, income_type을 x데이터, 직업을 y라벨로 해서 머신러닝 학습 후 예측 분류값을 채워 넣는 식으로 진행

**occyp_type 결측치 채우기**

```python
# income_type의 value 리스트
income_list = list(set(data_df['income_type'].values))

#income_type별로 가장 많은 occyp_type값을 리스트로 만듬
occyp_list = []
for i in income_list:
    occyp_list.append(data_df[data_df['income_type']== i]['occyp_type'].value_counts().index[0])

# 직업 종류 결측치 income_type으로 채우기
for i in data_df[data_df['occyp_type'].isnull()].index:
    if data_df['DAYS_EMPLOYED'][i] > 0:
        data_df['occyp_type'][i] = 'Unemployed'
    else:
        for income in income_list:
            if income == data_df['income_type'][i]:
                data_df['occyp_type'][i] = data_df[data_df['income_type']== income]['occyp_type'].value_counts().index[0]
data_df['occyp_type']
```