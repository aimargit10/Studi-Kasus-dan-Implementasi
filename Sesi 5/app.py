import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_data():
    """Membuat dataset demo yang relevan untuk dashboard interaktif."""
    df = px.data.gapminder().copy()
    df["gdp_per_capita"] = df["gdpPercap"]
    df["population_millions"] = df["pop"] / 1_000_000
    df["year"] = df["year"].astype(int)
    return df


def apply_filters(df: pd.DataFrame):
    """Menerapkan filter berdasarkan negara, benua, tahun, dan rentang populasi."""
    st.sidebar.header("Filter Data")

    continents = sorted(df["continent"].dropna().unique().tolist())
    selected_continents = st.sidebar.multiselect(
        "Pilih benua:",
        continents,
        default=continents,
    )

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    year_range = st.sidebar.slider(
        "Rentang tahun:",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1,
    )

    population_min, population_max = float(df["population_millions"].min()), float(df["population_millions"].max())
    population_range = st.sidebar.slider(
        "Rentang populasi (juta):",
        min_value=round(population_min, 2),
        max_value=round(population_max, 2),
        value=(round(population_min, 2), round(population_max, 2)),
        step=0.1,
    )

    filtered = df[
        (df["continent"].isin(selected_continents))
        & (df["year"].between(year_range[0], year_range[1]))
        & (df["population_millions"].between(population_range[0], population_range[1]))
    ].copy()

    return filtered


def main():
    """Dashboard utama dengan prinsip overview first, zoom/filter, details on demand."""
    st.set_page_config(page_title="Dashboard Interaktif", page_icon="📊", layout="wide")
    st.title("Dashboard Visualisasi Data Global")
    st.caption("Overview first, zoom & filter, details on demand")

    df = load_data()
    filtered = apply_filters(df)

    if filtered.empty:
        st.warning("Tidak ada data yang cocok dengan filter yang dipilih.")
        return

    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah negara unik", filtered["country"].nunique())
    col2.metric("Rata-rata harapan hidup", round(filtered["lifeExp"].mean(), 2))
    col3.metric("Rata-rata PDB per kapita", round(filtered["gdp_per_capita"].mean(), 2))

    fig_scatter = px.scatter(
        filtered,
        x="gdp_per_capita",
        y="lifeExp",
        size="population_millions",
        color="continent",
        hover_name="country",
        animation_frame="year",
        range_x=[filtered["gdp_per_capita"].min() * 0.8, filtered["gdp_per_capita"].max() * 1.2],
        range_y=[filtered["lifeExp"].min() * 0.9, filtered["lifeExp"].max() * 1.05],
        template="plotly_white",
        title="Hubungan GDP per Kapita vs Harapan Hidup",
    )
    fig_scatter.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color="white")))
    fig_scatter.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

    col_left, col_right = st.columns(2)

    fig_bar = px.bar(
        filtered.groupby(["continent", "year"], as_index=False)["lifeExp"].mean(),
        x="year",
        y="lifeExp",
        color="continent",
        barmode="group",
        template="plotly_white",
        title="Rata-rata Harapan Hidup per Benua",
    )
    col_left.plotly_chart(fig_bar, use_container_width=True)

    fig_line = px.line(
        filtered.groupby(["year", "continent"], as_index=False)["population_millions"].sum(),
        x="year",
        y="population_millions",
        color="continent",
        markers=True,
        template="plotly_white",
        title="Total Populasi per Benua",
    )
    col_right.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Details on Demand")
    st.dataframe(filtered.sort_values(["year", "country"]).reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()
