import os
import torch
import numpy as np
import rasterio
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import cv2
import folium
from folium.features import GeoJson
from rasterio.warp import transform
from huggingface_hub import snapshot_download
import traceback

# ==============================================================================
# 1. 설정: EuroSAT (Sentinel-2) 데이터로 학습한 최종 모델 (존재 확인 완료)
# ==============================================================================
# ⭐️ 100% 공개된, Sentinel-2 전문가 모델 (인증 필요 없음)
MODEL_NAME = "Chandanab/mit-b0-finetuned-eurosat"

# ⭐️ 이 모델의 경작지 관련 클래스 ID
# ID 0: AnnualCrop (일년생 작물)
# ID 5: Pasture (목초지)
# ID 6: PermanentCrop (다년생 작물)
CROP_CLASS_IDS = [0, 5, 6]

# 사용자 파일 설정
TIFF_FILE_PATH = 'data/2025-09-10, daedong_hs.data.tif'
OUTPUT_HTML_PATH = 'classification_map_Eurosat_Final.html'
# ⭐️ 새 모델을 위해 새 폴더에 저장합니다.
MODEL_SAVE_DIRECTORY = "./model_eurosat_chandanab/"
MIN_AREA_THRESHOLD = 150

# ==============================================================================
# 2. 모델 다운로드 기능 (인증 절대 필요 없음)
# ==============================================================================
def download_model_if_needed(repo_id, save_dir):
    """지정된 폴더에 모델이 없으면 자동으로 다운로드합니다."""
    if not os.path.exists(os.path.join(save_dir, "config.json")):
        print(f"'{repo_id}' 모델이 로컬에 없습니다.")
        print("모델 다운로드를 시작합니다. (100% 공개 모델, 인증 필요 없음)")
        try:
            # 이 모델은 100% 공개 모델이므로 인증 문제가 없습니다.
            snapshot_download(repo_id=repo_id, local_dir=save_dir, local_dir_use_symlinks=False, resume_download=True)
            print("🎉 모델 다운로드 성공!")
        except Exception as e:
            print(f"❌ 오류: 모델 다운로드 실패. {e}"); return False
    else:
        print(f"'{repo_id}' 모델이 로컬에 이미 존재합니다.")
    return True

# ==============================================================================
# 메인 분석 로직 시작 (이전과 동일)
# ==============================================================================
print("🤖 [EuroSAT 위성 전문 모델] 경작지 자동 분류를 시작합니다.")

# 단계 1: 모델 다운로드 또는 확인
if not download_model_if_needed(MODEL_NAME, MODEL_SAVE_DIRECTORY):
    exit(1)

# 단계 2: 모델 및 이미지 프로세서 불러오기
print("\n>> 단계 1: 로컬 폴더에서 모델을 불러옵니다...")
try:
    # ⭐️ AutoModelForSemanticEstimation -> AutoModelForSemanticSegmentation
    #    이전 코드의 오타를 수정했습니다.
    image_processor = AutoImageProcessor.from_pretrained(MODEL_SAVE_DIRECTORY)
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_SAVE_DIRECTORY)
    print("   ...모델 로딩 성공!")
except Exception:
    print(f"   ❌ 오류: 모델을 불러오는 중 문제가 발생했습니다."); traceback.print_exc(); exit(1)

# 단계 3: 위성 이미지 불러오기 및 전처리 (RGB 3밴드 사용)
print(f"\n>> 단계 2: 위성 이미지 '{TIFF_FILE_PATH}'를 불러오고 전처리합니다...")
try:
    with rasterio.open(TIFF_FILE_PATH) as src:
        bands = src.read([3, 2, 1]) # RGB 밴드 사용
        nodata_val = src.nodatavals[0]
        stretched_bands = []
        for band_data in bands:
            if nodata_val is not None:
                masked_data = np.ma.masked_equal(band_data, nodata_val)
                p2, p98 = np.percentile(masked_data.compressed(), (2, 98))
            else:
                p2, p98 = np.percentile(band_data, (2, 98))
            if p98 - p2 == 0: stretched = np.zeros_like(band_data, dtype=np.uint8)
            else:
                stretched = np.clip((band_data - p2) / (p98 - p2), 0, 1)
                stretched = (stretched * 255).astype(np.uint8)
            stretched_bands.append(stretched)
        rgb_image_np = np.dstack(stretched_bands)
        image = Image.fromarray(rgb_image_np)
        src_crs = src.crs
        src_transform = src.transform
        src_bounds = src.bounds
        print("   ...이미지 전처리 성공!")
except Exception:
    print(f"   ❌ 오류: 이미지 처리 중 문제가 발생했습니다."); traceback.print_exc(); exit(1)

# 단계 4: 모델 추론으로 경작지 예측
print("\n>> 단계 3: 모델을 사용하여 경작지 영역을 예측합니다...")
inputs = image_processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits.cpu()
upsampled_logits = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
pred_seg = upsampled_logits.argmax(dim=1)[0].numpy().astype(np.uint8)
print("   ...경작지 예측 완료!")

# 단계 5: 결과 후처리 및 지도 생성
print("\n>> 단계 4: 예측 결과를 분석하고 HTML 지도로 만듭니다...")
final_cropland_mask = np.isin(pred_seg, CROP_CLASS_IDS)
final_cropland_mask_uint8 = np.where(final_cropland_mask, 255, 0).astype(np.uint8)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_cropland_mask_uint8, connectivity=8)
center_lon = (src_bounds.left + src_bounds.right) / 2
center_lat = (src_bounds.top + src_bounds.bottom) / 2
center_coords_web = transform(src_crs, {'init': 'epsg:4326'}, [center_lon], [center_lat])
m = folium.Map(location=[center_coords_web[1][0], center_coords_web[0][0]], zoom_start=15, tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google Satellite')
final_field_count = 0
geojson_features = []
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA_THRESHOLD:
        final_field_count += 1
        component = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            coords = []
            for point in contour:
                lon, lat = src_transform * (point[0][0], point[0][1])
                lon_web, lat_web = transform(src_crs, {'init': 'epsg:4326'}, [lon], [lat])
                coords.append((lon_web[0], lat_web[0]))
            feature = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [coords]}, "properties": {"field_id": final_field_count}}
            geojson_features.append(feature)
if geojson_features:
    geojson_layer = GeoJson({"type": "FeatureCollection", "features": geojson_features}, style_function=lambda x: {'fillColor': '#4CAF50', 'color': '#2E7D32', 'weight': 2, 'fillOpacity': 0.6}, tooltip=folium.GeoJsonTooltip(fields=['field_id'], aliases=['경작지 번호:'])).add_to(m)
    m.fit_bounds(geojson_layer.get_bounds())
m.save(OUTPUT_HTML_PATH)
print("   ...지도 생성 완료!")
print("\n" + "="*50)
print(f"🎉 모든 작업 완료! 총 {final_field_count}개의 경작지를 식별하여 '{OUTPUT_HTML_PATH}' 파일로 저장했습니다.")
print("생성된 HTML 파일을 웹 브라우저에서 열어 결과를 확인해보세요.")
print("="*50)