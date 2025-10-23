import rasterio
import pandas as pd
from rasterio.warp import transform_geom
import tqdm
import requests
import time
import json

# --- 설정 ---
TIFF_FILE_PATH = 'data/2025-09-10, daedong_hs.data.tif'
OUTPUT_CSV_PATH = 'sampled_pixel_address.csv'  # 결과 파일 이름 변경

# ⭐️⭐️⭐️ 중요: 여기에 발급받은 V-World API 키를 입력하세요! ⭐️⭐️⭐️
VWORLD_API_KEY = "D4E1B616-546F-3016-8B28-9C76E4724FC4"

# ⭐️ 샘플링 간격 설정 (이전과 동일)
SAMPLING_INTERVAL = 50
API_CALL_DELAY = 0.1  # API 호출 간 최소 지연 시간 (초)

# V-World 리버스 지오코딩 API URL (사용자 예제 기반)
REVERSE_GEOCODING_URL = "https://api.vworld.kr/req/address"

print(f"'{TIFF_FILE_PATH}' 파일에서 {SAMPLING_INTERVAL}x{SAMPLING_INTERVAL} 간격으로 픽셀을 샘플링하여 주소로 변환합니다.")


# --- V-World API 호출 함수 (좌표 -> 주소, 사용자 예제 적용) ---
def get_address_from_coords_sampled(lon, lat, api_key):
    """V-World getAddress API를 호출하여 좌표로 주소를 가져옵니다."""
    params = {
        'service': 'address',
        'request': 'getAddress',  # 사용자 예제 API 요청 이름
        'version': '2.0',  # API 버전을 명시하는 것이 좋습니다.
        'crs': 'epsg:4326',  # 입력 좌표계 (WGS84 경위도)
        'point': f'{lon},{lat}',  # 경도, 위도 순서 확인 (V-World 문서 기준)
        'format': 'json',
        'type': 'both',  # 지번(parcel)과 도로명(road) 주소 모두 요청
        'zipcode': 'true',  # 우편번호 포함 여부 (선택 사항)
        'simple': 'false',  # 상세 주소 정보 요청 (선택 사항)
        'key': api_key
    }
    try:
        response = requests.get(REVERSE_GEOCODING_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # API 응답 상태 확인
        if data['response']['status'] == 'OK':
            # 주소 정보가 여러 개 반환될 수 있으므로 첫 번째 결과 사용
            # 응답 구조는 이전과 동일할 것으로 예상
            result = data['response']['result'][0]
            address_parcel = result.get('parcel', {}).get('text', '')  # 지번 주소
            address_road = result.get('road', {}).get('text', '')  # 도로명 주소
            return address_parcel, address_road
        else:
            # 'NOT_FOUND' 등 API 내부 오류 상태 출력
            # print(f"\n[API 응답 정보] 좌표 ({lon:.5f}, {lat:.5f}): Status = {data['response']['status']}")
            return None, None

    except requests.exceptions.Timeout:
        print(f"\n[API 호출 오류] 좌표 ({lon:.5f}, {lat:.5f}): Timeout 발생")
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"\n[API 호출 오류] 좌표 ({lon:.5f}, {lat:.5f}): {e}")
        return None, None
    except Exception as e:
        print(f"\n[처리 오류] 좌표 ({lon:.5f}, {lat:.5f}): {e}")
        try:
            print(f"  원본 응답 텍스트: {response.text}")
        except NameError:
            pass
        return None


# --- 메인 로직 (샘플링 적용, PNU 대신 주소 저장) ---
if "여기에_발급받은" in VWORLD_API_KEY:
    print("\n🚨 중요: 코드에 V-World API 키를 입력하지 않았습니다! API 키를 확인해주세요.")
    exit(1)

coordinates_data = []

with rasterio.open(TIFF_FILE_PATH) as src:
    src_crs = src.crs
    dst_crs = 'EPSG:4326'

    total_samples = (src.height // SAMPLING_INTERVAL + 1) * (src.width // SAMPLING_INTERVAL + 1)
    pbar = tqdm.tqdm(total=total_samples, desc="Processing sampled pixels")

    for row in range(0, src.height, SAMPLING_INTERVAL):
        for col in range(0, src.width, SAMPLING_INTERVAL):
            x, y = src.transform * (col, row)
            geom = {'type': 'Point', 'coordinates': (x, y)}
            transformed_geom = transform_geom(src_crs, dst_crs, geom)
            lon, lat = transformed_geom['coordinates']

            # V-World API 호출하여 주소 가져오기 (변경된 함수 사용)
            addr_parcel, addr_road = get_address_from_coords_sampled(lon, lat, VWORLD_API_KEY)

            # CSV에 저장할 데이터 구성 (PNU 대신 주소 컬럼 사용)
            coordinates_data.append({
                'pixel_col': col,
                'pixel_row': row,
                'longitude': lon,
                'latitude': lat,
                'address_parcel': addr_parcel if addr_parcel else '',
                'address_road': addr_road if addr_road else ''
            })

            pbar.update(1)
            time.sleep(API_CALL_DELAY)

    pbar.close()

# --- 결과 저장 ---
print("\n>> CSV 파일로 저장 중입니다...")
df = pd.DataFrame(coordinates_data)
df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

print("\n" + "=" * 50)
print(f"🎉 성공! 샘플링된 {len(coordinates_data)}개 픽셀의 좌표와 주소를 '{OUTPUT_CSV_PATH}' 파일로 저장했습니다.")
print(f"(샘플링 간격: {SAMPLING_INTERVAL} 픽셀)")
print("CSV 파일을 열어 주소가 제대로 변환되었는지 확인해보세요.")
print("=" * 50)