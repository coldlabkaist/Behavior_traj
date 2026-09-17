# 데이터 준비와 영상 보정

이 도구는 SiMBA에 넣을 tracking·영상 쌍을 준비합니다.
논문 그림·통계는 별도의 [analysis](../analysis/README.md)에서 기존 예측을 읽어 계산합니다.

- `tracking.py`: 파일 짝 찾기, 이름 정리, 좌표 단위 변환, 선택한 bodypart 제거, 누락 행 보완.
- `video.py`: 순차 디코딩과 임의 프레임 접근의 일치 검사, 프레임 대응 진단, 영상 재인코딩.
- `__main__.py`: 입력·출력 경로와 실행 옵션, 처리 보고서.

## 실행

아래 명령은 `direct_interaction/`에서 실행합니다. 입력 기본값은 `../data/csv/Fig5D_FigS9AB/tracking`, `../data/video/Fig5D_FigS9AB`입니다.
Python 3.9.25와 [requirements.txt](requirements.txt)의 환경에서 검증했습니다.
현재 컴퓨터에서는 `C:/Users/User/anaconda3/envs/moval/python.exe`를 사용합니다.

변환할 내용을 확인하고 보고서만 저장하려면:

```powershell
python -B -m preprocessing prepare --dry-run --output work/prepare_check
```

새 데이터의 CSV·영상을 준비하려면:

```powershell
python -B -m preprocessing prepare --csv-dir INPUT_CSV --video-dir INPUT_VIDEO --output work/prepared
```

기본 처리는 track 0·1, Neck 제거, 필요시 영상 해상도로 좌표 변환,
track별 선형 보간, 영상의 순차 디코딩 후 XVID 재인코딩입니다.
추가한 행의 score는 0으로 설정합니다. 원래 입력 파일은 수정하지 않습니다.
`--drop-bodyparts ""`로 제거를 끄고, `--keep-original-names`로 파일명을 유지할 수 있습니다.
기본 파일명 정리는 predict 접두사·시간 접미사·끝의 `_cut`을 제거하며, 충돌하면 실행을 중단합니다.

현재 영상과 tracking의 프레임 대응을 진단하려면:

```powershell
python -B -m preprocessing videos --output work/video_check
```

순차 디코딩 타임라인을 유지한 새 영상 사본을 만들려면:

```powershell
python -B -m preprocessing videos --rewrite --output work/video_rewrite
```

재인코딩 기본값은 OpenCV입니다. 기존 FFmpeg 방식은 `--backend ffmpeg`로 선택합니다.
FFmpeg가 PATH에 없으면 `--ffmpeg EXE_PATH`를 지정합니다.
영상 프레임 수가 tracking과 맞지 않으면 자동으로 잘라 맞추지 않고 오류로 보고합니다.
진단 결과의 seek 범위는 불일치를 설명하는 정보이며, 순차 영상에서 그 프레임을 추가 삭제하지 않습니다.

출력은 `direct_interaction/work/` 아래의 새 폴더에만 생성합니다.
`prepare`는 `csv/`, `video/`, `report.csv`를, 영상 보정은 `video/`, `report.csv`를 만듭니다.
진단과 dry-run은 보고서만 저장합니다. `--limit 1`로 한 쌍만 확인할 수 있습니다.
오류·추가 확인이 필요한 결과가 있으면 실행 종료 코드가 1입니다.

긴 형식의 `track, frame_idx` CSV와 SiMBA의 3행 헤더 pose CSV 모두 영상 진단에 사용할 수 있습니다.
좌표 준비는 긴 형식 CSV를 입력으로 받습니다.
