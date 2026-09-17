# Homecage 실행 안내

> 코드 전용 저장소입니다. 아래에 언급된 원자료·Final 결과·모델은 Git에 포함되지 않습니다.
> 실행 및 대조에 필요한 별도 자료를 안내된 경로에 배치한 뒤 실행하세요.

모든 명령은 `Behavior_traj/Homecage/`에서 실행합니다.

## 환경 준비

Python 3.9, PyTorch 2.1.2, R 4.4.1을 기준으로 합니다.

```powershell
cd Homecage
python -m pip install -r requirements.txt
Rscript analysis/run/install_r_packages.R
```

R 패키지 버전은 [R 통계 안내](analysis/core/statistics/README.md)를 참고하세요.
`Rscript`가 PATH에 없으면 실행 파일의 전체 경로를 사용합니다.

## 필요한 파일

| 작업 | 입력 |
|---|---|
| 논문 수치·그림 재현 | `analysis/output/Final/` 전체와 `checkpoints/best_clean.pth` |
| 원자료에서 latent 추출 | `../data/csv/Fig4B-G_FigS3-S7/{cont,experiments,roi}/`, checkpoint, `cfg/train_config.yaml`, `cfg/animal.yaml` |
| DAE 학습 | `../data/csv/Fig4B-G_FigS3-S7/{reference,roi}/`, `cfg/train_config.yaml`, `cfg/animal.yaml` |
| Movie S3 생성 | 원본 AVI, 원자료 CSV, Final의 pose 데이터, `analysis/resources/movie_s3.json` |

<a id="reproduction"></a>
## 논문 수치·그림 재현

```powershell
python -B -m analysis.run.reproduce
python -B -m analysis.run.figures
```

수치 검증 결과는 `analysis/work/paper/`, PNG·SVG는 `analysis/work/paper_figures/`에 생성합니다.
Final 입력은 그대로 유지합니다. 최종 Illustrator 조립은 포함하지 않습니다.
R 경로가 다르면 `reproduce --rscript "Rscript 실행 파일 경로"`로 지정합니다.

기본 명령은 저장된 latent·모형을 사용합니다. MLP·S5·S3 probe·R 모형의 재적합 명령과
패널별 실행 옵션은 [상세 실행 안내](analysis/run/README.md)에 있습니다.

<a id="latent-extraction"></a>
## 원자료에서 latent 추출

```powershell
python -B -m analysis.run.extract_latent --device cpu
python -B -m analysis.run.reproduce --refit-projection
```

추출 파일은 `analysis/work/latent_extraction/`에 생성하고 Final latent와 대조합니다.
`--refit-projection`은 Final latent로 MLP를 다시 학습하고 최종 3차원 좌표와 대조합니다.
추출을 다시 실행하려면 비어 있는 `--output analysis/work/새폴더`를 지정합니다.

현재 환경에서 원자료 재추출은 Final과 metadata가 같지만 latent 값까지 완전히 일치하지는
않습니다(최대 절대차 약 0.00372). 논문 downstream 재현에는 저장된 Final latent를 사용하세요.

재현 조건은 코드에 설정되어 있습니다. 변경할 때는 다음 조건을 유지하세요.

- CPU, PyTorch 2.1.2, 스레드 4개, 추론 배치 256.
- Control → VPA 입력 순서, 파일·창 순서, shuffle=False, augmentation=False.
- 겹치지 않는 30프레임 창과 설정 파일의 ROI·품질검사·정규화.
- MLP seed 42, Control cage-week당 최대 300개, 12-fold 및 기존 early stopping.
- BOI 입력 좌표는 `%.10g` 형식으로 CSV 저장 후 다시 읽기.

## DAE 학습

```powershell
python train.py --config cfg/train_config.yaml
```

단일 프로세스로 실행합니다. 학습 출력 위치는 설정의 `training.checkpoint_dir`을 따릅니다.

## Movie S3

```powershell
python -B -m analysis.run.movie_s3 --video-root D:/data_vid --video-root D:/Moval_proj/hc_more/raw_videos
```

`--video-root`는 원본 AVI 폴더에 맞게 바꾸며, 여러 폴더는 검색 우선순위 순으로 지정합니다.
FFmpeg·ffprobe가 PATH에 필요하고, Arial 글꼴을 사용합니다(없으면 대체 글꼴).
`--check-only`를 붙이면 선택 장면과 입력 경로만 확인합니다.

[생성 코드](analysis/plots/movie_s3.py)는 [선택 정보](analysis/resources/movie_s3.json)의
40개 장면을 10개 세트로 합성합니다. 기본 출력은
`analysis/work/movie_s3/Movie_S3.mp4`이며, 30초·10 fps·3840×1200입니다.
기존 출력이 있으면 별도의 `--output` 경로를 지정합니다.

## 설정과 데이터 로더

`cfg/train_config.yaml`과 `cfg/animal.yaml`은 함께 사용합니다. 새 학습 출력은
`training.checkpoint_dir`를 따르며, 논문 재현은 `checkpoints/best_clean.pth`를 사용합니다.
`datasets/pose_io.py`는 CSV·ROI 읽기, `preprocessing.py`는 보간·품질검사,
`dataset.py`는 학습·추론용 30프레임 창을 담당합니다. 코드에서는
`from datasets import PoseReader, MousePoseDataset`으로 불러옵니다.

설정·checkpoint의 `data/cont`, `data/experiments`, `data/reference`, `data/roi`는 기존 자료 식별자입니다.
`datasets/paths.py`가 루트 원자료 폴더로 연결하므로 Final의 metadata를 바꿀 필요가 없습니다.
