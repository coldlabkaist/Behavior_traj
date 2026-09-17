# Direct interaction 논문 결과 재현

대상은 **Fig5D(Social), FigS9A(Attentive), FigS9B(Prosocial)**입니다.
별도로 제공받은 최종 자료를 `output/final`에 배치합니다. 이 저장소에는 포함되지 않습니다.

## 실행

Python 3.12.4와 [requirements.txt](requirements.txt)의 버전을 사용합니다.
기준 PNG는 Matplotlib 3.8.4로 생성됐습니다. 다른 버전에서는 글자·선의 렌더링이 달라질 수 있습니다.
SiMBA 분류기 실행 환경과 이 논문 분석 환경은 별개입니다.

아래 명령은 `direct_interaction/analysis/`에서 실행합니다.

```powershell
python -B -m run.reproduce
```

`../../data/csv/Fig5D_FigS9AB/predictions/{modelA,modelB}`의 예측 확률에서 개체별 값을 계산하고 그림·통계를 만듭니다.
기본 출력은 `work/direct_interaction/`입니다. 기존 결과가 있으면
`--output-dir work/새폴더`를 지정하세요. 입력·보존 결과는 덮어쓰지 않습니다.

공통 개체별 표에서 그림·통계만 다시 만들려면:

```powershell
python -B -m run.reproduce --input-csv output/final/Fig5D/data/individual_behavior_values.csv --output-dir work/from_final
```

현재 컴퓨터에서는 `C:/Users/User/anaconda3/python.exe`가 위 버전의 실행 환경입니다.

## 계산 조건

- A/B 모델, 확률 ≥0.5, B6 데이터 124개 관측·56마리, 30 fps.
- Attentive: Approach, Facing, Following.
- Prosocial: Nose-Head, Nose-Body, Nose-Anogenital, Mounting.
- Social: 위 7개 행동의 합집합. 두 범주의 duration이나 bout count를 더하지 않습니다.
- 범주별로 양성 프레임을 합친 뒤, 중간 음성 구간이 15프레임 미만이면 병합합니다.
  병합한 bout가 15프레임 이상인 경우만 duration·bout count에 포함합니다.
- 주차별 Control/VPA 양측 Welch t-test. 각 범주 안의 3주차×2지표, 총 6개 비교를 BH 보정합니다.
- 그림은 전체 성별, 평균±SEM·개체별 선, mean-focus, 9.4×4.2 inch, 300 dpi, PNG·SVG입니다.

## 코드

- [core/behavior.py](core/behavior.py): 예측 확률 결합, 범주 합집합, bout 계산, 개체별 입력 표.
- [core/metadata.py](core/metadata.py): 파일명에서 개체·조건·주령을 연결.
- [core/statistics.py](core/statistics.py): Welch·BH 검정, SEM, 관측 주차별 개체 수.
- [plots/trajectory.py](plots/trajectory.py): 세 범주의 개체별 궤적과 평균·SEM 그림.
- [plots/style.py](plots/style.py): 색상·글꼴·PNG/SVG 저장 공통 함수.
- [run/reproduce.py](run/reproduce.py): 논문 기본 조건, 실행 옵션과 출력 관리.
- [run/verify.py](run/verify.py): 재생성 결과를 보존한 최종 데이터·통계·PNG와 비교.

`core`와 `plots`에는 명령 실행기를 두지 않습니다. `run`에서 이 모듈들을 불러옵니다.
`../../data/csv/Fig5D_FigS9AB/predictions/`는 예측 입력, `output/final/`은 보존 결과, `work/`는 재생성 결과입니다.
현재 루트 `data/video/Fig5D_FigS9AB`, `data/csv/Fig5D_FigS9AB/{tracking,predictions}`를 보존 기준으로 사용합니다.
과거 전처리 전 CSV는 이 배포본의 재현 입력에 포함하지 않습니다.
예측 CSV에서 논문에 사용하는 7개 행동만 읽고, 중간 CSV 없이 개체별 값을 계산합니다.

재현 결과를 확인하려면 아래 명령을 실행합니다. PNG까지 같은 환경에서는 모든 항목이 통과합니다.

```powershell
python -B -m run.verify --output-dir work/from_final
```
