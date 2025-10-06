import rasterio
import numpy as np
from PIL import Image

# 분석할 TIF 파일 경로
TIFF_FILE_PATH = 'data/2025-09-10, daedong_hs.data.tif'
# 저장할 이미지 파일 이름
OUTPUT_IMAGE_PATH = 'data/true_color_preview.png'

try:
    # rasterio를 사용해 TIF 파일을 엽니다.
    with rasterio.open(TIFF_FILE_PATH) as src:

        print("=" * 50)
        print(f"'{TIFF_FILE_PATH}' 파일 메타데이터 분석 결과")
        print("=" * 50)

        # 1. 파일의 기본 정보 출력
        print(f"✔️ 이미지 크기 (너비x높이): {src.width} x {src.height} 픽셀")
        print(f"✔️ 총 밴드(레이어) 수: {src.count} 개")
        print(f"✔️ 데이터 타입: {src.dtypes[0]}")
        print(f"✔️ 좌표계 정보 (CRS): {src.crs}")

        print("\n" + "-" * 50 + "\n")

        # 2. 각 밴드의 상세 정보 출력
        print("밴드별 상세 정보:")
        # get_descriptions()는 파일에 따라 설명이 없을 수 있습니다.
        # 설명이 없는 경우를 대비해 기본 설명을 만듭니다.
        band_descriptions = src.descriptions
        if all(d is None for d in band_descriptions):
            band_descriptions = [f"Band {i + 1}" for i in range(src.count)]

        for i, (dtype, desc) in enumerate(zip(src.dtypes, band_descriptions)):
            print(f"  - 밴드 {i + 1}:")
            print(f"    - 설명: {desc}")
            print(f"    - 데이터 타입: {dtype}")

        print("\n" + "=" * 50)
        print("분석 완료!")


except FileNotFoundError:
    print(f"❌ 오류: '{TIFF_FILE_PATH}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
except Exception as e:
    print(f"❌ 오류: 파일을 읽는 중 문제가 발생했습니다. \n{e}")


try:
    with rasterio.open(TIFF_FILE_PATH) as src:
        #1. 메타데이터에서 확인한 Red, Green, Blue 데이터 읽어오기
        print(">> Red(3), Green(2), Blue(1) 밴드를 읽어옵니다.")
        bands = src.read([3, 2, 1])

        # 2. 명압 대비 스트레칭
        # 각 밴드의 상위/하위 2% 값을 기준으로 값의 범위를 0-255로 재조정하여 선명하게 만듬
        # 파일에 지정된 NoData 값을 가져옵니다. (없으면 None)
        nodata_val = src.nodatavals[0] # 밴드별로 같다고 가정

        stretched_bands = []
        for band_data in bands:
            if nodata_val is not None:
                masked_data = np.ma.masked_equal(band_data, nodata_val)
                p2, p98 = np.percentile(masked_data.compressed(), (2,98))

            else:
                p2, p98 = np.percentile(band_data, (2, 98))

            # 0으로 나누는 것을 방지
            if p98 - p2 == 0:
                stretched = np.zeros_like(band_data, dtype=np.uint8)
            else:
                # 유요한 통계치를 기반으로 0~1 사이로 스케일링

                stretched = (band_data - p2) / (p98 - p2)
                stretched = np.clip(stretched, 0, 1)
                #0~255 범위의 8비트 정수로 변환
                stretched = (stretched * 255).astype(np.uint8)

            stretched_bands.append(stretched)


        # 3. 3개의 밴드를 하나의 RGB 이미지로 결합
        print(">> 밴드들을 RGB 이미지로 결합합니다.")
        #numpy의 stack기능을 사용하여 (높이, 너비, 채널) 형태의 3D 배열로 만듭니다.
        rgb_image = np.dstack(stretched_bands)

        # 4. Pillow 라이브러리를 사용해 이미지 객체로 변환하고 파일을 저장
        image_to_save = Image.fromarray(rgb_image)
        image_to_save.save(OUTPUT_IMAGE_PATH)

        print("\n" + "=" * 50)
        print(f" 성공 ! True color 이미지를 '{OUTPUT_IMAGE_PATH}' 파일로 저장했습니다.")


except FileNotFoundError:
    print(f"오류 '{TIFF_FILE_PATH}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"오류 : 이미지 생성 중 문제가 발생했습니다. \n{e}")


































