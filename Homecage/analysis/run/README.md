# 상세 실행 안내

모든 명령은 `Homecage/`에서 실행합니다. 환경 설치, DAE 학습과 Movie S3는
[프로젝트 README](../../README.md)에 있습니다. R 패키지와 버전은
[R 환경 안내](../core/statistics/README.md)를 참고하세요.

## 기본 재현

```powershell
python -B -m analysis.run.reproduce
python -B -m analysis.run.figures
```

입력은 `analysis/output/Final`, DAE checkpoint는 `checkpoints`입니다.
수치 검증은 `analysis/work/paper`, PNG/SVG는 `analysis/work/paper_figures`에 저장합니다.
다른 출력 위치는 `--output`으로 지정합니다. Final은 덮어쓰지 않습니다.

- `reproduce`: BOI, GMM posterior, 전이·점유도, soft joint 연령 순열검정,
  S3 학습 이력, S4 요인분석, S5 저장 반복 결과, S6 변화량, Fig5G/I Spearman,
  Fig4EFG/S7의 저장 R 모형을 검증합니다.
- `figures`: 저장된 최종 입력에서 논문 패널을 그립니다. 최종 Illustrator 조립은 제외합니다.
- R 경로는 `--rscript`로 지정합니다. 기본값은 `C:/Program Files/R/R-4.4.1/bin/Rscript.exe`입니다.

필요한 그림만 생성하려면:

```powershell
python -B -m analysis.run.figures --panels Fig4EFG FigS7AB
```

## 재적합과 원자료에서 latent 추출

```powershell
python -B -m analysis.run.reproduce --refit-projection
python -B -m analysis.run.reproduce --refit-s5 --output analysis/work/s5_refit
python -B -m analysis.run.extract_latent
python -B -m analysis.run.probe
Rscript analysis/run/fit_models.R
Rscript analysis/run/refit_sex_df.R
```

- `--refit-projection`: Final의 64차원 latent에서 Control 12 cage별 제외 MLP를 재학습하고
  3차원 좌표와 fold 이력을 비교합니다. BOI 입력은 `%.10g` CSV 저장·재읽기를 거칩니다.
- `--refit-s5`: Final의 codebook과 shifted coordinates에서 GMM 반복 적합을 수행합니다.
  원자료에서 shifted latent를 다시 추출하는 단계는 포함하지 않습니다.
- `extract_latent`: 원자료와 최종 checkpoint를 이용해 batch 256, torch thread 4로 추출하고
  Final의 z·relation·metadata와 비교합니다. 기존 출력이 있으면 새로운 `--output`을 지정합니다.
- `probe`: S3B의 5-fold nonlinear probe를 재학습합니다.
- `fit_models.R`: BOI 이분산 후보 모형을 재적합하고 Final RDS·CSV와 비교합니다.
- `refit_sex_df.R`: 성별 endpoint의 근사 자유도를 seed 20260914, extra.iter 200으로 재계산합니다.

이 재적합 명령들은 기본 재현 실행에 자동 포함되지 않습니다.
수치 검증과 그림의 픽셀 동일성은 별개이며, Final을 재생성본으로 자동 교체하지 않습니다.

## 입력·코드 연결

| 패널 | 계산 모듈 | 그림 모듈 |
|---|---|---|
| Fig4B | latent_cache, projection | distribution |
| Fig4C/D | motifs, transitions | motifs, transitions |
| Fig4EFG | boi, statistics | boi |
| Fig5G/I | rank_tests | associations |
| FigS3AB | nonlinear_probe | validation |
| FigS4AB | factor_analysis | validation |
| FigS5AB | motif_stability, temporal_shift | stability |
| FigS6AB | transitions | transitions |
| FigS7AB | statistics | boi |

계산 모듈은 `analysis/core`, 그림 모듈은 `analysis/plots`에 있습니다.
`core/archive.py`는 Final 읽기와 무결성 검사, `core/verification.py`는 수치 대조,
`plots/panels.py`는 패널 연결을 담당합니다. 공통 색상·글꼴은 `plots/style.py`,
PNG/SVG 저장은 `plots/export.py`에서 관리합니다.

S5의 codebook, compact shifted coordinates와 cage별 반복 결과는 필수 입력입니다.
Final의 패널별 `data`와 공통 입력을 함께 보존하세요. Manifest의 과거 경로는 출처 기록입니다.
