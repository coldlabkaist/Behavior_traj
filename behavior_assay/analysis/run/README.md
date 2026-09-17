# 논문용 실행 명령

실험별 계산은 `analysis/core/{three_chamber,mom_pup,oft}/paper.py`, 그림 처리는 `analysis/plots/{three_chamber,mom_pup,oft}/paper.py`에 있습니다. 이 폴더는 실행·결과 비교·저장을 담당합니다. 예전 개별 CLI는 각 실험 하위 폴더로 옮겼고, 논문용 명령은 그대로 유지했습니다.

`behavior_assay/output/Final`의 최종 입력을 사용합니다. 출력은 `behavior_assay/output/reproduced`이며, Final의 채택 그림·통계를 덮어쓰지 않습니다. 예전 `output/3chamber`, `output/mom_pup`, `output/OFT` 폴더를 읽지 않습니다.

Behavior_traj 루트에서, 분석 패키지가 설치된 Python 환경으로 실행합니다. 현재 컴퓨터에서는 `moval` conda 환경으로 전체 실행을 확인했습니다.

```powershell
conda activate moval
python behavior_assay/analysis/run/reproduce.py
```

특정 패널만 실행하거나 통계만 검증할 수 있습니다.

```powershell
python behavior_assay/analysis/run/reproduce.py --panels Fig5F Fig5H
python behavior_assay/analysis/run/reproduce.py --mode stats
python behavior_assay/analysis/run/reproduce.py --panels FigS10A FigS10B --output-dir behavior_assay/output/oft_check
```

- `--mode all`이 기본값입니다. 통계를 재계산하고 Final과 비교한 뒤 그림을 만듭니다.
- `--mode figures`도 통계를 재계산·비교한 후 그림에 표시합니다. 통계 CSV의 중복 export만 생략합니다.
- 통계 비교가 실패하면 해당 실행을 중단합니다. 결과를 조용히 바꿔 채택하지 않습니다.
- `--final-dir`로 입력 위치를 바꿀 수 있습니다. 기본 경로는 파일 위치에서 계산하므로 현재 작업 디렉터리와 사용자 이름에 의존하지 않습니다.
- 결과의 `verification.json`에는 비교한 통계표와 그림의 재생성 범위가 기록됩니다.

## 입력

| 패널 | Final 입력 |
|---|---|
| Fig5B, FigS8A | `inputs/three_chamber/spatial_maps.npz`의 density, support, ROI, 세션 수 |
| Fig5C, FigS8B | 각 패널 `data/individual_preference.csv` |
| FigS8C, FigS8D | 각 패널 `data/individual_metrics.csv` |
| Fig5E | `data/occupancy_counts.npz`, `data/selected_sessions.csv` |
| Fig5F | `data/cage_proximity.csv`의 `pct_time_sustained_body_scale_proximity` |
| Fig5H | `data/cage_velocity.csv`의 `avg_velocity_mm_s` |
| FigS10A | `data/density_control.csv`, `data/density_vpa.csv` |
| FigS10B | `data/individual_metrics.csv` |

Fig5F는 1초 이상 bout의 시간 비율을 검사하고 사용합니다. Fig5F·H의 비교는 Holm 보정, FigS10B는 캡션의 보정 전 Welch P값을 사용합니다. Fig5E 검출 예시 PNG는 외부 편집 원본에서 가져온 그림이므로 복사하며, 점유도 지도는 count grid로 다시 그립니다.

## Three-chamber 수치 지도 복원

누락되어 있던 density grid를 raw tracking으로 복원하여 Final에 추가했습니다. 82개 세션의 유효 좌표 수와 네 그룹의 요약값을 기존 자료와 대조했습니다. NPZ가 있으면 일반 실행은 raw 또는 과거 preprocessing 파일을 읽지 않습니다.

NPZ를 다시 만들려면 다음 명령을 사용합니다. Raw tracking과 ROI pin 파일은 루트 `data/csv/Fig5BC_FigS8ABCD`에 있어야 합니다. Final manifest의 기존 경로는 `analysis/paths.py`에서 새 위치로 연결합니다. 임시 preprocessing은 작업이 끝나면 제거합니다.

```powershell
python behavior_assay/analysis/run/prepare_density.py
```

현재 실행 범위는 **최종 개체·케이지별 수치와 수치 지도 → 통계·그림**입니다. 모든 행동 지표의 raw-to-metric 일괄 재계산은 별도 단계입니다. 기존 세부 폴더의 CLI는 원래 분석용 옵션을 포함하며, 논문 재현의 공식 진입점은 이 `run/reproduce.py`입니다.

## 검증

13개 통계표를 재계산해 Final과 대조했고 11개 패널을 모두 출력했습니다. Fig5C·FigS8B·C·D의 PNG는 원본과 픽셀까지 일치했습니다. Three-chamber 지도의 색상 격자도 원본 SVG의 래스터 내용과 일치합니다. 다른 그림에는 저장 시기·렌더링 환경에 따른 폰트/여백/래스터 차이가 있어 픽셀 동일성을 주장하지 않습니다. 채택된 Final 그림은 그대로 유지합니다.

프로젝트 밖의 작업 디렉터리에서도 전체 실행을 수행했고, 파일 읽기 감시에서 과거 output 폴더 접근은 0건이었습니다. 이후 과거 결과 폴더 6개에 대한 접근을 차단한 상태에서도 통계표 13개와 11개 패널의 재생성을 확인한 뒤, 해당 폴더와 중복 재생성본을 삭제했습니다. `output/reproduced`는 실행할 때 다시 생성됩니다.

확인한 환경: Python 3.9.25, numpy 1.26.4, pandas 2.3.3, scipy 1.13.1, matplotlib 3.9.4, statsmodels 0.14.6, OpenCV 4.11.0, shapely 2.0.6 (`moval`).

Python 패키지: numpy, pandas, scipy, matplotlib, statsmodels, patsy, Pillow, opencv-python, shapely. Raw preprocessing에는 프로젝트의 `data_loader` 모듈도 사용됩니다. OpenCV DLL 로딩이 정상인 활성화된 환경에서 실행하세요.
