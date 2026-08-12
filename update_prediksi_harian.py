# ============================================================
# update_prediksi_harian.py
# Pipeline harian: tarik data terbaru -> hitung fitur -> prediksi
# Dijalankan otomatis via GitHub Actions tiap hari
# ============================================================
import os
import time
import pandas as pd
import numpy as np
import requests
import joblib
import geopandas as gpd
from datetime import datetime, timedelta

# ------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------
# FIRMS_MAP_KEY dibaca dari environment variable, BUKAN hardcode -
# supaya aman karena repo ini PUBLIC di GitHub. Nilai aslinya
# disimpan di GitHub Secrets, otomatis tersedia sebagai env var
# saat GitHub Actions menjalankan workflow.
FIRMS_MAP_KEY = os.environ["FIRMS_MAP_KEY"]

# PATH RELATIF - mengikuti struktur folder repo GitHub:
#   FireRiskID/
#   ├── update_prediksi_harian.py   <- script ini
#   ├── data/
#   │   ├── kecamatan_centroid.csv
#   │   ├── histori_cuaca_kecamatan.csv
#   │   ├── histori_hotspot_kecamatan.csv
#   │   └── shapefile.zip            <- geopandas baca langsung dari .zip
#   └── model/
#       ├── model_firerisk_final.pkl
#       ├── kategori_info.pkl
#       └── threshold_info.pkl
#
# CATATAN PENTING beda dari versi Kaggle: di GitHub Actions TIDAK
# ADA lagi bedanya folder "read-only" vs "writable" seperti di
# Kaggle (/kaggle/input vs /kaggle/working). Seluruh repo di-checkout
# jadi satu folder biasa yang writable. Jadi TIDAK PERLU LAGI pola
# "_SUMBER" + "pastikan_writable()" yang dipakai di versi Kaggle -
# baca dan tulis langsung ke path yang sama di dalam repo. Setelah
# ditulis, workflow YAML yang akan commit perubahan itu balik ke
# repo (lihat update_harian.yml).
FILE_HISTORI_CUACA = "data/histori_cuaca_kecamatan.csv"
FILE_HISTORI_HOTSPOT = "data/histori_hotspot_kecamatan.csv"
FILE_CENTROID = "data/kecamatan_centroid.csv"
FILE_SHAPEFILE = "data/shapefile.zip"
FILE_MODEL = "model/model_firerisk_final.pkl"
FILE_KATEGORI_INFO = "model/kategori_info.pkl"
FILE_THRESHOLD_INFO = "model/threshold_info.pkl"
FILE_OUTPUT = "data/prediksi_kecamatan_terbaru.csv"

TANGGAL_HARI_INI = datetime.now().date()
TANGGAL_KEMARIN = TANGGAL_HARI_INI - timedelta(days=1)

# ------------------------------------------------------------
# Helper - normalisasi kolom teks, WAJIB dipanggil di semua
# titik yang baca/gabung data, biar konsisten (pelajaran dari
# insiden rate limit & mismatch casing sebelumnya)
# ------------------------------------------------------------
def normalisasi_kolom_wilayah(df):
    for col in ["provinsi", "Kab_Kota", "Kecamatan"]:
        if col in df.columns:
            df[col] = df[col].str.upper()
    return df

# ------------------------------------------------------------
# 0. PASTIKAN FILE HISTORI CUACA ADA DI WORKING DIR (writable)
#    Kalau belum ada (run pertama kali di sesi ini), copy dulu
#    dari dataset input sebagai titik awal. Run-run berikutnya
#    akan langsung pakai & update versi di working dir ini.
#
#    CATATAN GitHub Actions: file ini dibaca & ditulis langsung di
#    dalam repo (tidak ada lagi masalah read-only/writable seperti
#    di Kaggle). Workflow YAML yang meng-commit perubahan file ini
#    balik ke repo setelah script selesai jalan (lihat update_harian.yml).
# ------------------------------------------------------------

# ------------------------------------------------------------
# 1. TARIK CUACA HARI BARU (forecast Open-Meteo, bukan archive)
#    Dipanggil per kecamatan dengan JEDA antar request - belajar
#    dari insiden rate limit 429 di endpoint archive sebelumnya,
#    DAN dari insiden read timeout yang baru kejadian di endpoint
#    forecast ini (belum pernah dites sebelumnya - ternyata juga
#    perlu jeda kalau dipanggil beruntun untuk ratusan kecamatan
#    tanpa jeda sama sekali)
# ------------------------------------------------------------
JEDA_ANTAR_REQUEST = 0.5  # detik - mulai dari nilai kecil dulu

def tarik_cuaca_forecast(lat, lon, tanggal_mulai, tanggal_akhir):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": tanggal_mulai.isoformat(),
        "end_date": tanggal_akhir.isoformat(),
        "daily": "precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean",
        "timezone": "Asia/Jakarta"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["daily"]

def update_histori_cuaca():
    histori = pd.read_csv(FILE_HISTORI_CUACA, parse_dates=["tanggal"])
    histori = normalisasi_kolom_wilayah(histori)


    centroid = pd.read_csv(FILE_CENTROID)
    centroid = normalisasi_kolom_wilayah(centroid)

    tanggal_terakhir = histori["tanggal"].max().date()
    print(f"Tanggal terakhir di histori: {tanggal_terakhir}")

    if tanggal_terakhir >= TANGGAL_KEMARIN:
        print("Histori cuaca sudah up to date, tidak ada tanggal baru.")
        return histori

    tanggal_mulai = tanggal_terakhir + timedelta(days=1)

    baris_baru = []
    gagal = []
    for i, (_, row) in enumerate(centroid.iterrows()):
        try:
            data = tarik_cuaca_forecast(row["lat"], row["lon"], tanggal_mulai, TANGGAL_KEMARIN)
            for j, tgl in enumerate(data["time"]):
                baris_baru.append({
                    "provinsi": row["provinsi"],
                    "Kab_Kota": row["Kab_Kota"],
                    "Kecamatan": row["Kecamatan"],
                    "tanggal": tgl,
                    "curah_hujan": data["precipitation_sum"][j],
                    "kelembaban": data["relative_humidity_2m_mean"][j],
                    "suhu": data["temperature_2m_mean"][j],
                })
        except Exception as e:
            gagal.append((row["Kecamatan"], str(e)))
            print(f"Gagal tarik cuaca {row['Kecamatan']}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  ...progress: {i+1}/{len(centroid)} kecamatan")

        time.sleep(JEDA_ANTAR_REQUEST)

    print(f"Kecamatan gagal ditarik: {len(gagal)} dari {len(centroid)}")

    df_baru = pd.DataFrame(baris_baru)
    if len(df_baru) > 0:
        df_baru["tanggal"] = pd.to_datetime(df_baru["tanggal"])
        df_baru = normalisasi_kolom_wilayah(df_baru)
        histori = pd.concat([histori, df_baru], ignore_index=True)
        histori = histori.drop_duplicates(subset=["provinsi", "Kab_Kota", "Kecamatan", "tanggal"])
        histori.to_csv(FILE_HISTORI_CUACA, index=False)
        print(f"Histori cuaca ditambah {len(df_baru)} baris baru.")

    return histori

# ------------------------------------------------------------
# 2. TARIK HOTSPOT TERBARU (FIRMS NRT) + spatial join ke kecamatan
# ------------------------------------------------------------
def tarik_hotspot_nrt():
    # Bbox gabungan 3 provinsi (west, south, east, north) - sudah
    # terbukti berhasil di test manual. Endpoint "area" butuh bbox,
    # BUKAN kode negara (itu untuk endpoint "country" yang gagal
    # terus dengan 400 Bad Request meski format & MAP_KEY valid)
    BBOX = "100.0,-5.0,115.5,2.5"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/VIIRS_SNPP_NRT/{BBOX}/1"
    df = pd.read_csv(url)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    return df

def join_hotspot_ke_kecamatan(df_hotspot, gdf_kecamatan):
    gdf_kecamatan = gdf_kecamatan.copy()
    gdf_kecamatan = normalisasi_kolom_wilayah(
        gdf_kecamatan.rename(columns={"Provinsi": "provinsi"})
    )
    gdf_hotspot = gpd.GeoDataFrame(
        df_hotspot,
        geometry=gpd.points_from_xy(df_hotspot.longitude, df_hotspot.latitude),
        crs="EPSG:4326"
    )
    hasil = gpd.sjoin(gdf_hotspot, gdf_kecamatan, how="inner", predicate="within")
    return hasil

# ------------------------------------------------------------
# 2b. UPDATE HISTORI HOTSPOT HARIAN (archive permanen, sama
#     konsepnya dengan histori cuaca - terus bertambah tiap hari,
#     dipakai untuk hitung fitur lag hotspot_kemarin/2hari/3hari)
#
#     PENTING: hotspot_join di sini isinya titik-titik individual
#     hari INI SAJA (dari FIRMS NRT 1 hari terakhir), jadi diagregasi
#     dulu per kecamatan sebelum ditambahkan ke histori - supaya
#     formatnya konsisten dengan seed awal (1 baris = 1 kecamatan
#     + 1 tanggal + jumlah_hotspot, bukan 1 baris per titik).
#
#     Kecamatan yang TIDAK ada hotspot hari ini juga diisi 0 secara
#     eksplisit (sama seperti prinsip di Tahap 3 dulu) - supaya
#     fitur lag nanti tidak salah kaprah menganggap "tidak tercatat"
#     sebagai "tidak ada data" alih-alih "nol kejadian".
# ------------------------------------------------------------
def update_histori_hotspot(hotspot_join, centroid):
    histori = pd.read_csv(FILE_HISTORI_HOTSPOT, parse_dates=["tanggal"])
    histori = normalisasi_kolom_wilayah(histori)

    tanggal_terakhir = histori["tanggal"].max().date()
    if tanggal_terakhir >= TANGGAL_KEMARIN:
        print("Histori hotspot sudah up to date, tidak ada tanggal baru.")
        return histori

    agg_hari_ini = hotspot_join.groupby(
        ["provinsi", "Kab_Kota", "Kecamatan"]
    ).size().reset_index(name="jumlah_hotspot")
    agg_hari_ini = normalisasi_kolom_wilayah(agg_hari_ini)

    # Gabung ke SEMUA kecamatan (centroid = daftar lengkap 536),
    # supaya yang tidak ada hotspot hari ini eksplisit tercatat 0
    baris_baru = centroid[["provinsi", "Kab_Kota", "Kecamatan"]].merge(
        agg_hari_ini, on=["provinsi", "Kab_Kota", "Kecamatan"], how="left"
    )
    baris_baru["jumlah_hotspot"] = baris_baru["jumlah_hotspot"].fillna(0).astype(int)
    baris_baru["tanggal"] = pd.Timestamp(TANGGAL_KEMARIN)

    histori = pd.concat([histori, baris_baru], ignore_index=True)
    histori = histori.drop_duplicates(subset=["provinsi", "Kab_Kota", "Kecamatan", "tanggal"], keep="last")
    histori.to_csv(FILE_HISTORI_HOTSPOT, index=False)
    print(f"Histori hotspot ditambah {len(baris_baru)} baris baru (tanggal {TANGGAL_KEMARIN}).")

    return histori

# ------------------------------------------------------------
# 3. HITUNG FITUR HARI INI (dari buffer histori cuaca + hotspot)
#    hotspot_kemarin/2hari_lalu/3hari_lalu SEKARANG dihitung dari
#    histori_hotspot_kecamatan.csv beneran (sebelumnya hardcode 0)
# ------------------------------------------------------------
def hitung_fitur_hari_ini(histori_cuaca, histori_hotspot, centroid):
    batas_bawah_cuaca = TANGGAL_HARI_INI - timedelta(days=8)
    buffer_cuaca = histori_cuaca[histori_cuaca["tanggal"].dt.date >= batas_bawah_cuaca].copy()

    batas_bawah_hotspot = TANGGAL_HARI_INI - timedelta(days=4)
    buffer_hotspot = histori_hotspot[histori_hotspot["tanggal"].dt.date >= batas_bawah_hotspot].copy()

    hasil_fitur = []
    for _, row in centroid.iterrows():
        key = (row["provinsi"], row["Kab_Kota"], row["Kecamatan"])
        cuaca_kec = buffer_cuaca[
            (buffer_cuaca["provinsi"] == key[0]) &
            (buffer_cuaca["Kab_Kota"] == key[1]) &
            (buffer_cuaca["Kecamatan"] == key[2])
        ].sort_values("tanggal")

        if len(cuaca_kec) == 0:
            continue

        curah_hujan_avg_7d = cuaca_kec["curah_hujan"].tail(7).mean()
        suhu_avg_7d = cuaca_kec["suhu"].tail(7).mean()
        kelembaban_avg_7d = cuaca_kec["kelembaban"].tail(7).mean()

        curah_urut = cuaca_kec["curah_hujan"].tolist()
        hari_kering = 0
        for c in reversed(curah_urut):
            if c < 1.0:
                hari_kering += 1
            else:
                break

        # Ambil hotspot_kemarin, 2hari_lalu, 3hari_lalu dari histori
        # beneran - dicari per tanggal spesifik (TANGGAL_KEMARIN,
        # TANGGAL_KEMARIN-1, TANGGAL_KEMARIN-2), bukan hardcode 0
        hotspot_kec = buffer_hotspot[
            (buffer_hotspot["provinsi"] == key[0]) &
            (buffer_hotspot["Kab_Kota"] == key[1]) &
            (buffer_hotspot["Kecamatan"] == key[2])
        ]

        def ambil_hotspot_tanggal(tgl):
            baris = hotspot_kec[hotspot_kec["tanggal"].dt.date == tgl]
            return int(baris["jumlah_hotspot"].values[0]) if len(baris) else 0

        hotspot_kemarin = ambil_hotspot_tanggal(TANGGAL_KEMARIN)
        hotspot_2hari_lalu = ambil_hotspot_tanggal(TANGGAL_KEMARIN - timedelta(days=1))
        hotspot_3hari_lalu = ambil_hotspot_tanggal(TANGGAL_KEMARIN - timedelta(days=2))

        hasil_fitur.append({
            "provinsi": key[0], "Kab_Kota": key[1], "Kecamatan": key[2],
            "hotspot_kemarin": hotspot_kemarin,
            "hotspot_2hari_lalu": hotspot_2hari_lalu,
            "hotspot_3hari_lalu": hotspot_3hari_lalu,
            "curah_hujan_avg_7d": curah_hujan_avg_7d,
            "suhu_avg_7d": suhu_avg_7d,
            "kelembaban_avg_7d": kelembaban_avg_7d,
            "hari_tanpa_hujan": hari_kering,
            "bulan": TANGGAL_HARI_INI.month,
            "hari_dalam_tahun": TANGGAL_HARI_INI.timetuple().tm_yday,
        })

    return pd.DataFrame(hasil_fitur)

# ------------------------------------------------------------
# 4. PREDIKSI
#    PENTING: kategori_info.pkl masih pakai format lama untuk
#    kolom provinsi ("Riau", bukan "RIAU") - jadi provinsi
#    di-title-case-kan lagi khusus sebelum masuk model
#
#    CATATAN SOAL KECAMATAN YANG TIDAK DIKENAL MODEL:
#    28 dari 536 kecamatan (per 8 Agustus 2026) tidak pernah
#    tercatat hotspot sama sekali di data training 2020-2025
#    (kebanyakan kecamatan pusat kota: Palembang, Pekanbaru, dst),
#    jadi tidak pernah ikut proses training model. Kecamatan ini
#    TIDAK dilewatkan ke model (akan gagal encoding kategorikal),
#    melainkan diberi default "Rendah" secara eksplisit dengan
#    flag "model_terlatih=False", supaya dashboard/user bisa
#    membedakan mana hasil model beneran vs default historis.
# ------------------------------------------------------------
def prediksi(df_fitur):
    model = joblib.load(FILE_MODEL)
    kategori_info = joblib.load(FILE_KATEGORI_INFO)
    threshold_info = joblib.load(FILE_THRESHOLD_INFO)

    df_fitur = df_fitur.copy()

    # Pisahkan dulu: kecamatan yang dikenal vs tidak dikenal model,
    # SEBELUM proses encoding kategorikal (supaya yang tidak dikenal
    # tidak ikut coba-coba masuk model dan gagal diam-diam jadi NaN)
    kecamatan_dikenal = set(kategori_info["Kecamatan"])
    mask_dikenal = df_fitur["Kecamatan"].isin(kecamatan_dikenal)

    df_dikenal = df_fitur[mask_dikenal].copy()
    df_tidak_dikenal = df_fitur[~mask_dikenal].copy()

    if len(df_tidak_dikenal) > 0:
        print(f"INFO: {len(df_tidak_dikenal)} kecamatan tidak dikenal model "
              f"(tidak ada histori hotspot di data training) -> diberi default Rendah")
        print(f"  Daftar: {sorted(df_tidak_dikenal['Kecamatan'].tolist())}")

    # --- Proses kecamatan yang dikenal model (jalur normal) ---
    if len(df_dikenal) > 0:
        df_dikenal["provinsi"] = df_dikenal["provinsi"].str.title()
        # Kab_Kota & Kecamatan TIDAK diubah - training data sudah UPPERCASE utk kolom ini

        for col, kategori_valid in kategori_info.items():
            df_dikenal[col] = pd.Categorical(df_dikenal[col], categories=kategori_valid)

        # Cek ada baris lain yang gagal mapping (provinsi/Kab_Kota) - WAJIB dicek
        for col in kategori_info.keys():
            jumlah_nan = df_dikenal[col].isna().sum()
            if jumlah_nan > 0:
                print(f"PERINGATAN: {jumlah_nan} baris punya nilai '{col}' yang tidak dikenali model!")

        fitur_kolom = list(kategori_info.keys()) + [
            "hotspot_kemarin", "hotspot_2hari_lalu", "hotspot_3hari_lalu",
            "curah_hujan_avg_7d", "suhu_avg_7d", "kelembaban_avg_7d",
            "hari_tanpa_hujan", "bulan", "hari_dalam_tahun"
        ]

        df_dikenal["skor_risiko"] = model.predict_proba(df_dikenal[fitur_kolom])[:, 1]

        def kategorikan(skor):
            if skor >= threshold_info["CUTOFF_TINGGI"]:
                return "Tinggi"
            elif skor >= threshold_info["CUTOFF_SEDANG"]:
                return "Sedang"
            return "Rendah"

        df_dikenal["kategori_risiko"] = df_dikenal["skor_risiko"].apply(kategorikan)
        df_dikenal["model_terlatih"] = True

    # --- Proses kecamatan yang TIDAK dikenal model (default) ---
    if len(df_tidak_dikenal) > 0:
        df_tidak_dikenal["skor_risiko"] = np.nan  # tidak ada skor dari model
        df_tidak_dikenal["kategori_risiko"] = "Rendah"
        df_tidak_dikenal["model_terlatih"] = False

    # Gabung balik
    hasil = pd.concat([df_dikenal, df_tidak_dikenal], ignore_index=True)
    hasil["tanggal_prediksi"] = TANGGAL_HARI_INI
    return hasil

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    print(f"=== Update prediksi harian - {TANGGAL_HARI_INI} ===")

    centroid = pd.read_csv(FILE_CENTROID)
    centroid = normalisasi_kolom_wilayah(centroid)

    gdf_kecamatan = gpd.read_file(FILE_SHAPEFILE)

    histori_cuaca = update_histori_cuaca()

    print("Menarik hotspot NRT...")
    hotspot = tarik_hotspot_nrt()
    hotspot_join = join_hotspot_ke_kecamatan(hotspot, gdf_kecamatan)

    print("Meng-update histori hotspot harian...")
    histori_hotspot = update_histori_hotspot(hotspot_join, centroid)

    print("Menghitung fitur...")
    df_fitur = hitung_fitur_hari_ini(histori_cuaca, histori_hotspot, centroid)
    print(f"Jumlah kecamatan dengan fitur lengkap: {len(df_fitur)} dari {len(centroid)}")

    print("Menjalankan prediksi...")
    hasil = prediksi(df_fitur)

    hasil.to_csv(FILE_OUTPUT, index=False)
    print(f"\nSelesai. Prediksi tersimpan ke {FILE_OUTPUT}")
    print(hasil["kategori_risiko"].value_counts())

# ============================================================
# CATATAN: CARA SET FIRMS_MAP_KEY SAAT TESTING LOKAL/MANUAL
# ============================================================
# Script ini baca dari os.environ["FIRMS_MAP_KEY"]. Untuk testing
# manual di komputer lokal (bukan lewat GitHub Actions), set dulu
# environment variable ini sebelum run, contoh (Linux/Mac):
#
#   export FIRMS_MAP_KEY="dc22344f15d7523b008e97a4784472f1"
#   python update_prediksi_harian.py
#
# atau di Windows PowerShell:
#
#   $env:FIRMS_MAP_KEY = "dc22344f15d7523b008e97a4784472f1"
#   python update_prediksi_harian.py
#
# JANGAN PERNAH menulis key asli langsung di file script ini atau
# di file manapun yang ikut ter-commit ke repo GitHub public.
#
# Untuk GitHub Actions (otomasi harian yang sebenarnya), key ini
# otomatis tersedia sebagai environment variable dari GitHub Secrets
# - tidak perlu di-set manual, lihat update_harian.yml.
#
