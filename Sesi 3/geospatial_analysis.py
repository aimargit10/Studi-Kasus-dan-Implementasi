import json
from pathlib import Path

import geopandas as gpd
import folium
import matplotlib.pyplot as plt


def create_dummy_geojson(output_path: str | Path) -> Path:
    """Membuat file GeoJSON contoh dengan beberapa wilayah administratif dummy."""
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Wilayah A",
                    "population": 120000,
                    "region": "Utara",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [106.7, -6.1],
                            [106.9, -6.1],
                            [106.9, -6.3],
                            [106.7, -6.3],
                            [106.7, -6.1],
                        ]
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "name": "Wilayah B",
                    "population": 220000,
                    "region": "Selatan",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [106.9, -6.1],
                            [107.1, -6.1],
                            [107.1, -6.3],
                            [106.9, -6.3],
                            [106.9, -6.1],
                        ]
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "name": "Wilayah C",
                    "population": 500000,
                    "region": "Tengah",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [106.8, -6.0],
                            [107.0, -6.0],
                            [107.0, -6.2],
                            [106.8, -6.2],
                            [106.8, -6.0],
                        ]
                    ],
                },
            },
        ],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        json.dump(data, file)

    return output


def load_spatial_data(file_path: str | Path) -> gpd.GeoDataFrame:
    """Memuat data spasial dari file GeoJSON ke GeoDataFrame."""
    return gpd.read_file(file_path)


def filter_by_attribute(gdf: gpd.GeoDataFrame, min_population: int = 0, region: str | None = None) -> gpd.GeoDataFrame:
    """Melakukan filter berdasarkan atribut populasi dan/atau nama region."""
    filtered = gdf.copy()
    filtered = filtered[filtered["population"] >= min_population]

    if region:
        filtered = filtered[filtered["region"] == region]

    return filtered.reset_index(drop=True)


def create_interactive_map(gdf: gpd.GeoDataFrame, output_html: str | Path) -> str:
    """Membuat peta interaktif menggunakan Folium dan menyimpannya sebagai HTML."""
    centroid = gdf.geometry.unary_union.centroid
    map_obj = folium.Map(location=[centroid.y, centroid.x], zoom_start=10, tiles="OpenStreetMap")

    folium.GeoJson(
        gdf.to_json(),
        style_function=lambda feature: {
            "fillColor": "#4CAF50",
            "color": "#2E7D32",
            "weight": 2,
            "fillOpacity": 0.5,
        },
        tooltip=folium.GeoJsonTooltip(fields=["name", "population", "region"], aliases=["Wilayah", "Populasi", "Region"]),
    ).add_to(map_obj)

    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(output)
    return str(output)


def create_static_map(gdf: gpd.GeoDataFrame, output_png: str | Path) -> str:
    """Membuat peta statis menggunakan Matplotlib lalu menyimpannya sebagai PNG."""
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(ax=ax, column="population", cmap="viridis", edgecolor="black", linewidth=1.2)
    ax.set_title("Peta Wilayah Administratif")
    ax.set_axis_off()

    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def main() -> None:
    """Eksekusi pipeline utama geospatial analysis."""
    geojson_path = Path(__file__).resolve().parent / "dummy_administrative_data.geojson"
    html_output = Path(__file__).resolve().parent / "geospatial_map.html"
    png_output = Path(__file__).resolve().parent / "geospatial_map.png"

    # Langkah 1: buat file contoh jika belum ada.
    create_dummy_geojson(geojson_path)

    # Langkah 2: baca data spasial dari GeoJSON.
    gdf = load_spatial_data(geojson_path)
    print("Data spasial terload:")
    print(gdf[["name", "population", "region"]])

    # Langkah 3: filter berdasarkan atribut.
    filtered = filter_by_attribute(gdf, min_population=150000, region="Utara")
    print("\nHasil filter spatial:")
    print(filtered[["name", "population", "region"]])

    # Langkah 4: simpan hasil ke peta interaktif dan statis.
    html_path = create_interactive_map(filtered, html_output)
    png_path = create_static_map(filtered, png_output)

    print(f"\nPeta interaktif disimpan di: {html_path}")
    print(f"Peta statis disimpan di: {png_path}")


if __name__ == "__main__":
    main()
