# Direct interaction

> 코드 전용 저장소입니다. 아래에 언급된 원자료·Final 결과·모델은 Git에 포함되지 않습니다.
> 실행 및 대조에 필요한 별도 자료를 안내된 경로에 배치한 뒤 실행하세요.

SiMBA 예측 확률에서 직접 상호작용 지표를 계산합니다. 논문 패널은 Fig5D와 FigS9A–B입니다.

- `../data/csv/Fig5D_FigS9AB/tracking`: tracking CSV
- `../data/csv/Fig5D_FigS9AB/predictions/{modelA,modelB}`: 분석에 쓰는 SiMBA 예측 CSV
- `../data/video/Fig5D_FigS9AB`: 원영상, 요청 시 제공
- `modelA`, `modelB`: SiMBA 프로젝트·모델, 요청 시 제공
- `analysis/core`, `plots`, `run`: 계산, 그림, 실행 명령
- `analysis/output/final`: 채택한 데이터·통계·PNG/SVG
- `preprocessing`: tracking 준비와 영상 보정 도구

분석 환경과 실행 방법은 [분석 안내](analysis/README.md), 데이터 준비는
[전처리 안내](preprocessing/README.md)를 따릅니다. SiMBA 분류기 학습 환경은 별도입니다.

프로젝트 최상위에서:

```powershell
cd direct_interaction/analysis
python -m pip install -r requirements.txt
python -B -m run.reproduce
python -B -m run.verify
```

재생성 결과는 `analysis/work/direct_interaction`에 저장합니다. 기존 출력이 있으면
다른 `--output-dir work/폴더명`을 지정합니다. 비교 후 작업 결과는 삭제해도 됩니다.
