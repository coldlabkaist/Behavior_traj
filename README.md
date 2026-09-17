# MovAl behavior analyses

논문에 사용한 분석·플롯·전처리·학습 코드와 설정, 실행 안내를 네 프로젝트로 관리합니다.

**코드 전용 저장소입니다. 원자료·분석 데이터·통계·그림·latent·학습 모델은 포함하지 않습니다.**
실행에 필요한 자료는 별도로 받아 아래 경로에 배치해야 합니다.

| 프로젝트 | 분석 | 채택 결과 |
|---|---|---|
| [Homecage](Homecage/README.md) | DAE latent, motif, BOI | Fig4B–G, Fig5G·I, FigS3–S7 |
| [MovAl_benchmark](MovAl_benchmark/README.md) | Tracking miss, identity switch, jitter, RMSE | Fig2B–E, FigS1, Table S1–S2 |
| [direct_interaction](direct_interaction/README.md) | SiMBA 예측 기반 직접 상호작용 | Fig5D, FigS9A–B |
| [behavior_assay](behavior_assay/README.md) | Three-chamber, mother–pup, open field | Fig5B·C·E·F·H, FigS8, FigS10 |

별도 결과 묶음의 Table S1용 tracking-miss 통계 위치는 `MovAl_benchmark/Fig2B/stat`입니다.
최종 Illustrator 조립과 원고 편집은 이 실행 범위에 포함되지 않습니다.

## 구조와 재현

각 프로젝트는 `core`에서 계산하고 `plots`에서 그림을 만들며 `run`에서 실행합니다.
패널별 채택 자료는 `figure`, `stat`, `data`로 관리합니다. `work`와 `reproduced`는
재생성·검증 출력이며, 채택 자료에 필요한 입력으로 사용하지 않습니다.

각 README에 작업 디렉터리, 환경, 실행 명령과 재현 범위를 명시했습니다.
Homecage/behavior_assay는 Python 3.9, benchmark/direct_interaction은 Python 3.12에서
검증했습니다. 프로젝트별 requirements를 사용하며 서로 다른 환경을 한꺼번에 합치지 않습니다.
Homecage의 통계 검증에는 R도 필요합니다.

## 원자료 위치와 제공 범위

원자료는 저장소 루트의 `data/video`, `data/csv` 아래에 모읍니다.
여러 패널이 같은 원자료를 쓰면 공동 폴더를 사용합니다.

| 하위 폴더 | 원자료 |
|---|---|
| `Fig2BCD_FigS1` | 모델별 benchmark tracking |
| `Fig2E` | RMSE 정답·예측 좌표 |
| `Fig4B-G_FigS3-S7` | Homecage Control/VPA, DAE 학습 reference, ROI |
| `Fig5BC_FigS8ABCD` | Three-chamber tracking과 ROI |
| `Fig5D_FigS9AB` | Direct interaction tracking·SiMBA 예측·원영상 |
| `Fig5EFH` | Mother–pup tracking |
| `FigS10AB` | Open-field tracking |

현재 `video`에는 direct interaction 영상 151개가 있습니다. 다른 영상 폴더는 자료를
추가할 자리이며, 외부 경로의 영상은 자동 복사하지 않았습니다.
CSV 원자료 묶음을 받은 뒤 `data/csv/<피규어 폴더>` 구조로 풀면 기본 실행 경로에 연결됩니다.
`data/manifest.csv`는 2,064개 원자료 파일의 크기·SHA256·이전 위치를 기록합니다.

- Git 저장소: 분석·플롯·전처리·학습 코드, 설정과 실행 안내.
- 별도 재현 입력·결과 묶음: Final 분석 데이터·latent·통계·그림, 학습 모델과 checkpoint.
- 별도 데이터 묶음: raw tracking·ROI·정답 좌표·SiMBA 예측 CSV.
- 요청 시 제공: 원영상과 SiMBA 프로젝트·학습 모델.
- 로컬 작업용: R 패키지 설치 폴더, 캐시, 재생성 결과와 조사 기록.

루트 `data/`는 Git에서 제외합니다. Final의 분석 결과와 모델도 Git에서 제외하며, 로컬의 기존 패널 폴더에 보존합니다.
Manifest와 checkpoint의 과거 파일 경로는 자료 식별자이며, 현재 코드가 새 원자료 위치로
연결합니다. 원고와 Illustrator 조립은 자동으로 수정하지 않습니다.

## 통합 Git 저장소

`Behavior_traj` 전체를 하나의 저장소로 관리합니다. 네 프로젝트는 일반 하위 폴더이며,
별도 저장소나 submodule이 아닙니다. 제외 규칙은 루트 `.gitignore`에 모았습니다.

공개할 `main` 이력에는 코드·설정·실행 안내만 포함합니다. 데이터와 결과가 포함된 과거
로컬 이력은 업로드하지 않습니다. 이 코드 전용 저장소에는 Git LFS가 필요하지 않습니다.

## 별도 입력 배치

원자료 외에도 분석별로 다음 위치의 별도 입력·모형·대조 결과가 필요합니다.

- `Homecage/analysis/output/Final/`, `Homecage/checkpoints/`
- `MovAl_benchmark/Fig2B/`, `Fig2C/`, `Fig2D_FigS1_TableS2/`, `Fig2E/`
- `direct_interaction/analysis/output/final/`
- `behavior_assay/output/Final/`

저장소를 복제하는 것만으로 위 파일들이 제공되지는 않습니다. 각 프로젝트 README의
필수 입력을 배치한 뒤 실행합니다. 결과 대조 명령에는 별도로 제공되는 기준 결과도 필요합니다.
