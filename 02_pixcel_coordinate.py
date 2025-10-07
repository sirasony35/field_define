import rasterio
import pandas as pd
from rasterio.warp import transform_geom
import tqdm  # 진행 상황을 보여주기 위한 라이브러리

# --- 설정 ---
TIFF_FILE_PATH = 'data/2025-09-10, daedong_hs.data.tif'
OUTPUT_CSV_PATH = 'pixel_coordinates.csv'

print(f"'{TIFF_FILE_PATH}' 파일의 모든 픽셀 좌표를 추출합니다.")
print("이미지 크기가 크므로 시간이 다소 걸릴 수 있습니다...")

# 좌표 정보를 저장할 리스트
coordinates_data = []

with rasterio.open(TIFF_FILE_PATH) as src:
    # 원본 좌표계와 웹 지도에서 사용하는 표준 좌표계(EPSG:4326)
    src_crs = src.crs
    dst_crs = 'EPSG:4326'

    # tqdm을 사용하여 진행률 표시줄 생성
    for row in tqdm.tqdm(range(src.height), desc="Processing rows"):
        for col in range(src.width):
            # 1. 픽셀 좌표(col, row)를 원본 좌표계의 물리적 좌표로 변환
            x, y = src.transform * (col, row)

            # 2. 원본 좌표를 표준 경도, 위도 좌표로 변환
            geom = {'type': 'Point', 'coordinates': (x, y)}
            transformed_geom = transform_geom(src_crs, dst_crs, geom)
            lon, lat = transformed_geom['coordinates']

            coordinates_data.append({
                'pixel_col': col,
                'pixel_row': row,
                'longitude': lon,
                'latitude': lat
            })

# 리스트를 Pandas DataFrame으로 변환
print("\n>> CSV 파일로 저장 중입니다...")
df = pd.DataFrame(coordinates_data)

# CSV 파일로 저장
df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

print("\n" + "=" * 50)
print(f"🎉 성공! 모든 픽셀의 좌표를 '{OUTPUT_CSV_PATH}' 파일로 저장했습니다.")
print("=" * 50)