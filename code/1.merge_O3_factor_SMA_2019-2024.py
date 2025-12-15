import pandas as pd
import numpy as np
import os
from scipy.spatial import KDTree

os.chdir("D:/정리/코드 포토폴리오/LightGBM_O3_factor")
print(os.getcwd())

merged_df_ozone = pd.read_csv(
    "./data/SMA_O3_AirKorea_test.csv",
    encoding="utf-8"
)

merged_df = pd.read_csv(
    "./data/SMA_HCHO_NO2_TROPOMI_test.csv",
    encoding="utf-8"
)


merged_df_ozone['ozone'] = pd.to_numeric(
    merged_df_ozone['ozone'], errors='coerce'
)
merged_df_ozone['ozone'] = merged_df_ozone['ozone'].replace(-999.0, np.nan)
merged_df_ozone = merged_df_ozone.dropna(subset=["ozone"])


def find_closest_points_and_update_kdtree(df_ozone, df_FNR):
    updated_rows = []

    grouped_ozone = df_ozone.groupby(['month', 'day'])
    grouped_FNR   = df_FNR.groupby(['month', 'day'])

    for (month, day), ozone_group in grouped_ozone:
        if (month, day) in grouped_FNR.groups:
            fnr_group = grouped_FNR.get_group((month, day))

            tree = KDTree(fnr_group[['lat', 'lon']].values)

            distances, indices = tree.query(
                ozone_group[['lat', 'lon']].values
            )

            updated_group = ozone_group.copy()
            updated_group[['lat', 'lon']] = (
                fnr_group[['lat', 'lon']]
                .iloc[indices.flatten()]
                .values
            )

            updated_rows.append(updated_group)

    return pd.concat(updated_rows, ignore_index=True)

updated_df_ozone = find_closest_points_and_update_kdtree(
    merged_df_ozone,
    merged_df
)


updated_df_ozone['ozone'] = pd.to_numeric(
    updated_df_ozone['ozone'], errors='coerce'
)
updated_df_ozone['ozone'] = updated_df_ozone['ozone'].replace(-999.0, np.nan)

updated_df_ozone = (
    updated_df_ozone
    .groupby(
        ["site", "lon", "lat", "month", "day"],
        as_index=False
    )["ozone"]
    .mean()
)

merged_df11 = pd.merge(
    merged_df,
    updated_df_ozone,
    on=['lat', 'lon', 'month', 'day']
)[[
    "month", "day", "lat", "lon",
    "HCHO", "NO2", "ozone", "site"
]]

merged_df11['ozone'] = pd.to_numeric(
    merged_df11['ozone'], errors='coerce'
)
merged_df11['ozone'] = merged_df11['ozone'].replace(-999.0, np.nan)
merged_df11 = merged_df11.dropna(
    subset=['HCHO', 'NO2', 'ozone']
)


merged_df11['FNR'] = merged_df11['HCHO'] / merged_df11['NO2']
merged_df11.loc[merged_df11['FNR'] <= 0, 'FNR'] = np.nan

merged_df11['O3_unit'] = merged_df11['ozone'] * 1.96 * 1e3
merged_df11 = merged_df11.dropna(
    subset=['FNR', 'O3_unit']
)


ERA5_file_path = './data/1330_SMA_ERA5_test.csv'
all_df = pd.read_csv(ERA5_file_path)

merged_df11['month'] = merged_df11['month'].astype(int)
merged_df11['day']   = merged_df11['day'].astype(int)
merged_df11['site']  = merged_df11['site'].astype(int)

merged_df55 = pd.merge(
    merged_df11,
    all_df,
    on=['site', 'month', 'day'],
    how='inner'
)

merged_df55 = merged_df55.drop(
    columns='Unnamed: 0',
    errors='ignore'
)


save_path = "./mid_result/O3_factor_SMA.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

merged_df55.to_csv(save_path, index=False)
