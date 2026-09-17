# Preprocessing

Tracking CSV의 품질 요약과 Body_C 보간 도구입니다.

- `report_data.py`: 파일·개체별 결측 및 중복 프레임 요약
- `interpolate_body_c.py`: 중복 처리, 프레임 보충, Body_C 보간

`behavior_assay` 폴더에서 실행합니다.

```powershell
python -B -m preprocessing.report_data --help
python -B -m preprocessing.interpolate_body_c --help
```

논문 그림·통계의 기본 재현에는 [분석 실행기](../analysis/run/README.md)를 사용합니다.
