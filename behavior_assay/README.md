# Behavior assays

> 코드 전용 저장소입니다. 아래에 언급된 원자료·Final 결과·모델은 Git에 포함되지 않습니다.
> 실행 및 대조에 필요한 별도 자료를 안내된 경로에 배치한 뒤 실행하세요.

Three-chamber, mother–pup, open-field 실험의 논문 분석입니다.

| 실험 | 패널 |
|---|---|
| Three-chamber | Fig5B·C, FigS8A–D |
| Mother–pup | Fig5E·F·H |
| Open field | FigS10A·B |

- `../data/csv/Fig5BC_FigS8ABCD`, `Fig5EFH`, `FigS10AB`: 실험별 raw tracking과 ROI 입력
- `data_loader`, `preprocessing`: CSV 읽기·전처리
- `analysis/core`, `plots`, `run`: 실험별 계산, 그림, 실행 명령
- `analysis/tools`: ROI·접촉 검수 GUI
- `output/Final`: 패널별 figure·stat·data와 공통 필수 입력

프로젝트 최상위에서 Python 3.9 환경을 활성화한 뒤 실행합니다.

```powershell
python -m pip install -r behavior_assay/requirements.txt
python -B behavior_assay/analysis/run/reproduce.py
```

입력은 Final, 재생성 출력은 `output/reproduced`입니다. 재현 범위는
**최종 개체·케이지별 데이터와 수치 지도 → 통계·그림**이며, 모든 지표의 raw-to-metric
일괄 재계산을 의미하지 않습니다. 출력은 비교 후 삭제해도 됩니다.

[실행 옵션과 입력](analysis/run/README.md)을 참고하세요.
Fig5E의 검출 예시는 외부 편집 원본에서 가져온 그림을 사용합니다.
