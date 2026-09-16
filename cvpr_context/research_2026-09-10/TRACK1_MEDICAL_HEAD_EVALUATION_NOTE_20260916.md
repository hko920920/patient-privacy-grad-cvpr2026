# 실행 전 평가 해석 보완

새 head 특징 추출 중, 생성 실행·영상 점수 관측 전에 정했다. 최초 protocol과 비교군·seed·모델·표본 수를 변경하지 않는다.

미리 정한 혼합 normal/effusion32장 KID는 기존 `kid_unbiased` 식을 그대로 계산하되, 같은 latent가 두 prompt에 쓰이므로32장 IID 표본의 불편성까지 주장하지 않는다. **분포 기술 점수**로만 보고한다. 같은 latent 안의 연관이 결과 해석에 미치는 영향을 드러내기 위해 normal16 vs real20, effusion16 vs real20의 조건별 KID와 동일가중 평균을 함께 보고한다. 각 조건 안에서는 서로 다른16 latent를 사용한다. 상반된 점수 중 유리한 것만 선택하지 않는다.

PRDC 역시 작은 개발표본의 기술 통계다. BioViL cosine 및 margin의 pooled-public 차이는 동일 seed의 세 specific prompt 평균으로16개 block을 만들고, 고정 RNG26091631·4,000회 paired block bootstrap으로 기술 범위를 계산한다. 이 반복은 새 이미지가 아니며 환자 일반화·임상 성능·논문 성공 검정이 아니다. 공식적인 효용 판정은 여러 점수·전체 paired 영상과 함께 제한적으로 한다.

평가기 재현성은 원표 전체 재추론 대신 첫8개 real/generated 혼합 입력을 같은 batch 경로로 두 번 처리해 최대 drift≤1e-5를 요구한다. 기준을 넘으면 metric ranking을 해석하기 전에 구현/환경 차이를 조사한다. 저장된 전체 feature·text embedding으로 점수 산술을 독립 재계산한다. 임상 정확성은 평가하지 않는다.
