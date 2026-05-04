# tools logic
import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

# ── API keys from .env ─────────────────────────────────────────────────────────
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
EVENTBRITE_API_KEY = os.getenv("EVENTBRITE_API_KEY")


@tool
def get_chicago_weather(month: str) -> str:
    """
    Use this tool when the user asks about Chicago weather, what to pack,
    or what the weather will be like during their visit.
    Accepts a month name (e.g. 'June', 'December').
    Returns current Chicago weather from OpenWeatherMap plus seasonal
    packing tips for the requested month.
    Data source: OpenWeatherMap API - https://openweathermap.org/api
    """
    print("The agent is using get_chicago_weather tool")

    seasonal_tips = {
        "january": "Very cold. Wind chill can reach -20F. Heavy coat, thermal layers, gloves, scarf, waterproof boots.",
        "february": "Still very cold and snowy. Layer up heavily. Waterproof boots essential.",
        "march": "Cold transitioning to cool. Unpredictable weather. Medium-heavy coat plus waterproof layer.",
        "april": "Cool and rainy. Light to medium jacket and waterproof layer. Comfortable walking shoes.",
        "may": "Mild and pleasant. Light jacket for evenings. Great time to visit.",
        "june": "Warm and sunny. T-shirts and light layers. Evenings can be cool near the lake.",
        "july": "Hot and humid. Light clothing, sunscreen, sunglasses. Bring a light rain jacket.",
        "august": "Hot and humid with thunderstorms. Light clothing, sunscreen, rain jacket.",
        "september": "Warm and pleasant - one of the best months to visit. Light jacket for evenings.",
        "october": "Cool and crisp. Medium jacket, layers. Evenings get cold. Beautiful fall colors.",
        "november": "Cold and grey. Heavy jacket, scarf, gloves. Prepare for cold rain and wind.",
        "december": "Cold and snowy. Heavy coat, gloves, scarf, waterproof boots. Beautiful holiday season."
    }

    month_lower = month.lower().strip()

    if month_lower not in seasonal_tips:
        return f"Invalid month: '{month}'. Please provide a full month name like 'June'."

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        "?lat=41.8781&lon=-87.6298"
        f"&appid={OPENWEATHER_API_KEY}"
        "&units=imperial"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        current_temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"].capitalize()
        wind_speed = data["wind"]["speed"]
        return (
            f"Current Chicago weather: {current_temp}F (feels like {feels_like}F), "
            f"{description}, humidity {humidity}%, wind {wind_speed} mph.\n\n"
            f"Planning to visit in {month.capitalize()}?\n"
            f"Packing tips: {seasonal_tips[month_lower]}"
        )
    except Exception as e:
        return (
            f"Could not fetch live weather (error: {str(e)}).\n\n"
            f"Seasonal tips for {month.capitalize()}: {seasonal_tips[month_lower]}"
        )


@tool
def search_chicago_places(query: str) -> str:
    """
    Use this tool when the user asks for restaurant, attraction, hotel,
    or activity recommendations in Chicago.
    Accepts a search query (e.g. 'deep dish pizza near the Loop',
    'art museums in Chicago', 'rooftop bars River North').
    Returns top matching places with name, address, rating, and price level.
    Data source: Google Maps Places API - https://developers.google.com/maps/documentation/places/web-service
    """
    print("The agent is using search_chicago_places tool")

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": f"{query} Chicago", "key": GOOGLE_MAPS_API_KEY}

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("status") != "OK":
            return f"Places search error: {data.get('status')} - {data.get('error_message', 'No results found.')}"

        results = data.get("results", [])[:5]
        if not results:
            return "No places found for that search query."

        output = [f"Top Chicago results for '{query}':"]
        for place in results:
            name = place.get("name", "N/A")
            address = place.get("formatted_address", "N/A")
            rating = place.get("rating", "No rating")
            price = place.get("price_level", None)
            price_str = " | Price: " + "$" * price if price is not None else ""
            output.append(f"- {name} | {address} | Rating: {rating}{price_str}")
        return "\n".join(output)

    except Exception as e:
        return f"Places tool error: {str(e)}"


@tool
def get_chicago_events(date_range: str) -> str:
    """
    Use this tool when the user asks about events, festivals, concerts,
    or things happening in Chicago during their visit.
    Accepts a date range as a string (e.g. 'June 10 to June 13 2025').
    Returns upcoming Chicago events with name, date, venue, and ticket link.
    Data source: Eventbrite API - https://www.eventbrite.com/platform/api
    """
    print("The agent is using get_chicago_events tool")

    url = "https://www.eventbriteapi.com/v3/events/search/"
    headers = {"Authorization": f"Bearer {EVENTBRITE_API_KEY}"}
    params = {"location.address": "Chicago, IL", "location.within": "10mi",
              "q": "Chicago", "sort_by": "date", "expand": "venue"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        events = data.get("events", [])[:5]

        if not events:
            return f"No events found in Chicago for '{date_range}'. Try a broader date range."

        output = [f"Upcoming Chicago events around {date_range}:"]
        for event in events:
            name = event.get("name", {}).get("text", "N/A")
            start = event.get("start", {}).get("local", "N/A")
            venue = event.get("venue") or {}
            venue_name = venue.get("name", "Venue TBD")
            ticket_url = event.get("url", "")
            output.append(f"- {name} | {start} | {venue_name} | {ticket_url}")
        return "\n".join(output)

    except Exception as e:
        return f"Events tool error: {str(e)}"
