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
   TROPOMI overpass(현지 시각 : 13:30)에 맞춰 한국 SMA(서울,인천,경기도)에 대해 가공한 파일

참고 사항
1. 각 코드의 위 부분의 os.chdir() 에 자신의 경로로 수정해야함
2. 폰트는 제공하지 않음, 사용자의 설정 필요

프로젝트 구성
```text
LightGBM_O3_factor/
├─ README.md
│
├─ code/                                   # Python code
│  ├─ 1.merge_O3_factor_SMA_2019-2024.py
│  ├─ 2_0.LightGBM_hpo.py
│  └─ 2_1.LightGBM_SMA.py
│
├─ data/                                  
│  ├─ 1330_SMA_ERA5_test.csv
│  ├─ SMA_HCHO_NO2_TROPOMI_test.csv
│  └─ SMA_O3_AirKorea_test.csv
│
├─ mid_result/                                  # LightGBM의 하이퍼 파라미터 최적화 결과 및 중간 산출물 
│  ├─ GEMS_FNR_Threshold_SMA.txt
│  └─ LightGBM_hpo/
│      ├─ best_params.json
│      ├─ lightgbm_final_result_SMA.csv
│      └─ optuna_full_results_SMA.csv
│
├─ LightGBM_result                             # LightGBM의 SHAP, PDP 분석을 통한 오존 생성의 인위적, 기상적 요인 분석 및 모델 성능 평가 결과
│  ├─ pdp_HCHO_SMA.png
│  ├─ pdp_NO2_SMA.png
│  ├─ shap_summary_SMA.png
│  └─ test_performance_SMA.txt
