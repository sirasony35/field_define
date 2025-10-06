import torch
import numpy as np
import rasterio
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import cv2
import folium
from folium.features import GeoJson
from rasterio.warp import transform

print('경작지 자동분류를 시작합니다.')
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# 1. 설정: 파일 경로, 모델 이름 등 기본 정보를 정의합니다.
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
TIFF_FILE_PATH = 'data/2025-09-10, daedong_hs.data.tif'
OUTPUT_HTML_PATH = 'data/classification_map.html'
MODEL_NAME = "microsoft/beit-large-patch16-512-ade20k-semantic-segmentation"
CROPLAND_CLASS_ID = 13
MIN_AREA_THRESHOLD = 150

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# 2. 모델 및 이미지 프로세서 불러오기
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
print("\n>> 단계 1: Hugging Face에서 사전 학습 모델을 불러옵니다...")
try:
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
    print(' ...  모델 로딩 성공!')

except Exception as e:
    print(f" 오류: 모델 로딩 오류")
    exit()