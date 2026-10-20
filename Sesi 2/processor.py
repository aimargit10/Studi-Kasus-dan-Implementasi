from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd


logger = logging.getLogger(__name__)


def clean_weather_data(raw_data: Dict[str, Any]) -> pd.DataFrame:
    """Membersihkan payload JSON API menjadi DataFrame yang siap dianalisis."""
    if not isinstance(raw_data, dict):
        raise ValueError("Data mentah harus bertipe dictionary.")

    daily = raw_data.get("daily")
    if not isinstance(daily, dict):
        raise ValueError("Data JSON tidak memiliki bidang 'daily'.")

    required_keys = {"time", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"}
    missing_keys = required_keys - set(daily.keys())
    if missing_keys:
        raise ValueError(f"Kolom penting tidak ditemukan pada data harian: {sorted(missing_keys)}")

    df = pd.DataFrame(
        {
            "date": daily.get("time", []),
            "temperature_max_c": daily.get("temperature_2m_max", []),
            "temperature_min_c": daily.get("temperature_2m_min", []),
            "precipitation_sum_mm": daily.get("precipitation_sum", []),
            "wind_speed_max_kmh": daily.get("wind_speed_10m_max", []),
        }
    )

    # Ubah tipe data tanggal menjadi datetime agar analisis temporal lebih mudah.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Hapus baris dengan tanggal tidak valid.
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    # Pastikan kolom numerik diubah ke tipe numerik dan nilai kosong diisi dengan median.
    numeric_cols = [
        "temperature_max_c",
        "temperature_min_c",
        "precipitation_sum_mm",
        "wind_speed_max_kmh",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    logger.info("Data cuaca berhasil dibersihkan dan dipersiapkan ke tabel DataFrame.")
    return df


def summarize_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Menyusun ringkasan statistik dasar dari data cuaca."""
    if df.empty:
        return df

    summary = df.agg(
        {
            "temperature_max_c": ["min", "max", "mean"],
            "temperature_min_c": ["min", "max", "mean"],
            "precipitation_sum_mm": ["sum", "mean"],
            "wind_speed_max_kmh": ["max", "mean"],
        }
    )
    summary = summary.round(2)
    return summary


def save_to_csv(df: pd.DataFrame, output_path: str) -> str:
    """Menyimpan DataFrame ke file CSV lokal."""
    try:
        df.to_csv(output_path, index=False)
        logger.info("Data berhasil disimpan ke %s", output_path)
        return output_path
    except Exception as exc:
        logger.error("Gagal menyimpan file CSV: %s", exc)
        raise RuntimeError(f"Gagal menulis file CSV ke {output_path}.") from exc
