LightGBM을 이용해 오존 생성의 인위 및 기상 요인 분석

data 설명
1. SMA_O3_AirKorea_test.csv
   :오존 농도 자료
   AirKorea(https://www.airkorea.or.kr/) 의 지상 오존을 다운받은 후
   한국 SMA(서울,인천,경기도)에 대해 가공한 파일

2. SMA_HCHO_NO2_TROPOMI_test.csv
   :오존 생성의 인위적 원인(HCHO, NO2) 자료
   TROPOMI(https://www.earthdata.nasa.gov/) 의 HCHO, NO2를 다운받은 후
   한국 SMA(서울,인천,경기도)에 대해 가공하고 재격자화한 파일

3. 1330_SMA_ERA5_test.csv
   : 오존 생성의 기상적 요인(RH, SSR, u10, v10) 자료
   ERA5(https://cds.climate.copernicus.eu/) 의 hourly single level 자료 다운받은 후
   국 SMA(서울,인천,경기도)에 대해 가공한 파일

