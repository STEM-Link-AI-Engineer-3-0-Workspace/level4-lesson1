"""Shared weather code. Not a lesson concept.

Open-Meteo (https://open-meteo.com) is a free weather API. No account, no key,
no card. That is why we can use it live in a classroom.

Two endpoints are involved:

    geocoding-api.open-meteo.com/v1/search   a place name  -> a latitude/longitude
    api.open-meteo.com/v1/forecast           a lat/lon     -> the weather

04 writes its rainfall tool out in full, so you can see what a tool actually
does. 07 is about a different tool (one that sends an alert), so its forecast
tool, get_forecast, lives here instead.
"""

import requests
from langchain.tools import tool

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Colombo"


def geocode(place: str) -> tuple[float, float, str]:
    """Turn a place name into (latitude, longitude, the name the API matched).

    Returning the matched name matters: ask for "Polonnaruwa" and you get
    Polonnaruwa, but ask for something ambiguous and you want to know what it
    decided you meant.
    """
    results = requests.get(
        GEOCODE_URL,
        params={"name": place, "count": 1, "language": "en", "format": "json"},
        timeout=20,
    ).json().get("results")

    if not results:
        raise LookupError(f"Open-Meteo has no place called {place!r}")

    hit = results[0]
    return hit["latitude"], hit["longitude"], hit["name"]


@tool
def get_forecast(district: str) -> str:
    """Get the 48-hour rainfall forecast for a Sri Lankan district, in mm."""
    print(f">>> get_forecast: {district}")

    lat, lon, name = geocode(district)
    hourly = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation",
            "timezone": TIMEZONE,
            "forecast_hours": 48,     # the next 48 hours, starting now
        },
        timeout=20,
    ).json()["hourly"]

    rain = round(sum(hourly["precipitation"]), 1)
    return f"{name}: {rain} mm expected in the next 48 hours."
