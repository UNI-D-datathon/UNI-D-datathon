# UNI-D 4th Baseline
## Image Denoising Task Using Restormer

### 코드 구조

```
${PROJECT}
├── README.md
├── preprocess.py
├── model.py
├── train.py
└── test.py
```
- preprocess.py -> preprocess data : unzip후 데이터 경로를 알맞게 수정할 것
- test.py -> Inference : Image Denoising Inference 코드
- train.py -> Training Restormer 
- model.py -> Implementation of Restormer 

# Installation
## 가상 환경 생성
```bash
conda create -n image_resolution python=3.8
conda activate image_resolution
```

## 패키지 설치
```bash
git clone https://github.com/UNI-D-datathon/image-resolution
cd image-resolution
pip install -r requirements.txt
```

# 파일 실행
## train.py
```bash
python train.py
```
## test.py
```bash
python test.py
```