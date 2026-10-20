import logging
from pathlib import Path

from fetcher import fetch_weather_data
from processor import clean_weather_data, save_to_csv, summarize_weather_data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Pipeline utama untuk mengambil, membersihkan, dan menyimpan data cuaca."""
    latitude = -6.2088
    longitude = 106.8456
    start_date = "2024-01-01"
    end_date = "2024-01-30"
    output_path = Path(__file__).resolve().parent / "weather_data.csv"

    try:
        logger.info("Memulai pipeline pengambilan data cuaca...")
        raw_data = fetch_weather_data(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
        )

        cleaned_df = clean_weather_data(raw_data)
        summary_df = summarize_weather_data(cleaned_df)

        logger.info("Tabel data cuaca yang telah dibersihkan:")
        print(cleaned_df.head())

        logger.info("Ringkasan statistik data cuaca:")
        print(summary_df)

        save_to_csv(cleaned_df, str(output_path))
        logger.info("Pipeline selesai dengan sukses. File disimpan di: %s", output_path)

    except RuntimeError as exc:
        logger.error("Kesalahan selama eksekusi pipeline: %s", exc)
    except Exception as exc:
        logger.exception("Terjadi error tak terduga pada pipeline: %s", exc)


if __name__ == "__main__":
    main()
