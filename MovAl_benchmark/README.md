# MovAl tracking benchmark

> 코드 전용 저장소입니다. 아래에 언급된 원자료·Final 결과·모델은 Git에 포함되지 않습니다.
> 실행 및 대조에 필요한 별도 자료를 안내된 경로에 배치한 뒤 실행하세요.

Fig2B–E·FigS1·Table S1–S2의 tracking miss, identity switch, jitter, RMSE 분석입니다.

## 구조

- `analysis/core`: 입력 읽기, 집계, 대응 통계, FFT, 좌표 RMSE
- `analysis/plots`: 공통 스타일과 PNG/SVG 생성
- `analysis/run`: 재현 실행과 채택 결과 대조
- `Fig2B`, `Fig2C`, `Fig2D_FigS1_TableS2`, `Fig2E`: 패널별 figure·stat·data
- `../data/csv/Fig2BCD_FigS1/{Raw_*,SegCont_*}`: 모델별 raw tracking CSV
- `reference`: 조립 그림 참고본

`work`는 재생성 결과 및 로컬 조사 기록입니다. 공개 분석의 입력으로 사용하지 않습니다.

## 실행

프로젝트 최상위 `Behavior_traj`에서 Python 3.12 환경을 활성화한 뒤 실행합니다.

```powershell
python -m pip install -r MovAl_benchmark/analysis/requirements.txt
python -B -m MovAl_benchmark.analysis.run.reproduce
python -B -m MovAl_benchmark.analysis.run.verify
```

기본 출력은 `MovAl_benchmark/work/reproduced`입니다. 다른 위치를 쓰려면 두 명령 모두에
`--output-root <경로>`를 전달합니다. 재현 실행에 `--figure Fig2B`, `Fig2C`, `Fig2D`,
`FigS1`, `Fig2E`를 지정하면 해당 패널만 생성합니다. 전체 verify에는 전체 패널 출력이 필요합니다.

## 입력과 계산

### Fig2B · Table S1

`Fig2B/data/tracking_miss_long.csv`를 사용합니다. Raw·Seg·Seg-Cont 입력 중 논문에는
Raw·Seg-Cont를 사용합니다. 키포인트마다 세 개체를 먼저 평균한 영상 17개를 단위로
평균±SEM, 2×3 반복측정 ANOVA, 15개 대응 t 검정과 Holm 보정을 계산합니다.
Table S1용 통계는 `Fig2B/stat`에 있습니다. 과거 참고용 `saved_plot_summary.csv` 대신
현재 채택한 `plot_summary.csv`를 기준으로 합니다.

### Fig2C

`Fig2C/data/Mode_comp.xlsx`의 기존 실험 값을 사용합니다. 거리 분석용 재카운트로
교체하지 않습니다. 2×3 반복측정 ANOVA와 조건별 MovAl 대비 네 대응 t 검정을 수행하고
BH 보정합니다. 그림은 `figure/identity_switch_frequency.png`와 `.svg`입니다.

### Fig2D · FigS1 · Table S2

`../data/csv/Fig2BCD_FigS1` 아래의 Raw/SegCont SLEAP·MovAl 폴더의 CSV
68개를 사용합니다. 프레임 0–8999에서 세 track의 연속 위치 차이를 구하고,
유효한 양수 이동량의 log10 값에서 평균을 제거한 뒤 FFT를 적용합니다.
Low [0,0.05), Mid [0.05,0.15), High [0.15,0.5) 구간의 파워를 합합니다.

개체별 파워 1,836행, 영상별 파워 612행, ANOVA 27행, 대응 비교 54행을 생성합니다.
통계는 영상별 평균에서 계산하며 기존 플롯의 집계 정의를 유지합니다.
`Body_C`, `Nose`, `Tail`의 밴드 파워와 예시 이동 궤적을 함께 생성합니다.

### Fig2E

`../data/csv/Fig2E`의 정답 5개와 예측 30개 CSV에서 계산합니다. 프레임 번호를 보정하고
0 좌표를 결측 처리한 뒤 모델별 이미지 크기로 정규화합니다. 다섯 클립을 합쳐,
키포인트별 유효 예측·정답 좌표 쌍의 제곱거리를 평균하고 제곱근을 취합니다.
DLC 개체 라벨은 이름만 대응시키며 오차를 줄이기 위한 개체 재배정은 하지 않습니다.
중간 MSE는 반올림하지 않고 그림의 RMSE만 소수 네 자리로 표시합니다. 추론 통계는 없습니다.

## 검증과 보존

현재 경로에서 전체 재현·대조 18개 항목이 통과했습니다. Fig2B·C·E PNG는 채택 그림과
픽셀까지 일치합니다. Jitter는 저장된 수치와 계산 정의를 대조하며, 과거 여섯 조건으로
조립된 그림의 배치까지 복제하지 않습니다. 검증기는 불일치 시 종료 코드 1을 반환합니다.

원고 수정은 자동으로 수행하지 않습니다. 원고 수정 안내와 과거 비교 기록은 로컬
`work/manuscript_update`, `work/before_current_data_update` 등에 보관하며 공개 실행에는
필요하지 않습니다. 최종 그림·통계·원자료는 재생성 출력과 구분해 유지합니다.
