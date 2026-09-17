# R 통계 계산

이 파일들은 함수를 정의하며, 불러오기만 해서는 분석을 실행하지 않습니다.

| 파일 | 함수 | 계산 |
|---|---|---|
| `inference.R` | `verify_frozen_inference(out)` | 동결 BOI·성별 모형의 추론과 Final 통계 대조 |
| `fit_models.R` | `refit_models(out)` | BOI 공분산 후보·성별 모형 재적합 |
| `refit_sex_df.R` | `refit_sex_df(out)` | 성별 endpoint의 근사 자유도·대비 검정 재계산 |

명령행 실행은 `analysis/run`의 같은 이름의 R 파일을 사용합니다. 실행기는
인수와 출력 위치를 검사하고 필요한 라이브러리 경로를 설정한 다음 이 함수를 호출합니다.
프로젝트 루트인 `Homecage/`에서 실행합니다. 로컬 패키지가 있는 `.r-library/`를 먼저 사용하고,
그다음 R의 기본·사용자 라이브러리에서 패키지를 찾습니다.

## 패키지 준비

```powershell
Rscript analysis/run/install_r_packages.R
```

이 명령은 사용할 수 없는 패키지만 `.r-library/`에 설치합니다. 이미 있는 패키지는
업데이트하지 않습니다. 설치 폴더는 운영체제별 로컬 환경이므로 Git에서 제외합니다.

| 패키지 | 검증에 사용한 버전 | 역할 |
|---|---|---|
| nlme | 3.1.164 | 혼합효과 모형 |
| clubSandwich | 0.7.0 | 군집 보정 공분산·검정 |
| sandwich | 3.1.3 | 공분산 계산 의존성 |
| emmeans | 1.11.1 | 성별 대비·근사 자유도 |

검증에 사용한 R은 4.4.1입니다. 위 표는 검증 환경 기록이며, 설치 명령이 버전을
고정하지는 않습니다. 새 환경에서는 `analysis.run.reproduce`의 수치 대조를 실행합니다.
