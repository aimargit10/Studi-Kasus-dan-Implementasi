import logging
from typing import Any, Dict, Optional

import requests


logger = logging.getLogger(__name__)


def fetch_weather_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "auto",
    daily_params: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Mengambil data cuaca dari Open-Meteo API secara aman dan terstruktur.

    Args:
        latitude (float): Koordinat lintang lokasi.
        longitude (float): Koordinat bujur lokasi.
        start_date (str): Tanggal mulai format YYYY-MM-DD.
        end_date (str): Tanggal akhir format YYYY-MM-DD.
        timezone (str): Zona waktu API. Default 'auto'.
        daily_params (Optional[list[str]]): Parameter harian yang diinginkan.

    Returns:
        Dict[str, Any]: Data JSON mentah hasil API.

    Raises:
        RuntimeError: Jika permintaan gagal atau response tidak valid.
    """
    if daily_params is None:
        daily_params = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ]

    endpoint = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "daily": ",".join(daily_params),
    }

    try:
        response = requests.get(endpoint, params=params, timeout=20)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Waktu koneksi ke API habis (timeout).")
        raise RuntimeError("Gagal mengambil data cuaca: koneksi timeout.") from None
    except requests.exceptions.ConnectionError:
        logger.error("Koneksi jaringan gagal saat menghubungi API cuaca.")
        raise RuntimeError("Gagal mengambil data cuaca: koneksi jaringan gagal.") from None
    except requests.exceptions.HTTPError as exc:
        logger.error("Server API merespons error HTTP: %s", exc)
        raise RuntimeError(f"Gagal mengambil data cuaca: HTTP {response.status_code}.") from exc
    except requests.exceptions.RequestException as exc:
        logger.error("Permintaan ke API gagal: %s", exc)
        raise RuntimeError("Gagal mengambil data cuaca: permintaan gagal.") from exc

    try:
        payload = response.json()
    except ValueError:
        logger.error("Response API bukan format JSON yang valid.")
        raise RuntimeError("Gagal mengambil data cuaca: respons API tidak valid.") from None

    if "daily" not in payload:
        logger.error("Format response API tidak memuat data daily.")
        raise RuntimeError("Gagal mengambil data cuaca: format response tidak sesuai.")

    logger.info("Data cuaca berhasil diambil dari API Open-Meteo.")
    return payload
