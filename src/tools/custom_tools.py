"""
Custom LangChain tools.

Ported from Day 5 notebook (day5_react_agents_tools.ipynb).
Each tool is decorated with @tool so LangChain agents can discover and invoke them.
"""
from __future__ import annotations

import math

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.

    Use this for any arithmetic, algebra, or math calculations.
    Examples: '2 + 2', '42 * 7', 'sqrt(144)', '(10 + 5) * 3 / 2'
    """
    try:
        safe_globals = {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "pi": math.pi,
            "e": math.e,
            "sin": math.sin,
            "cos": math.cos,
            "log": math.log,
            "abs": abs,
            "round": round,
        }
        result = eval(expression, safe_globals)  # noqa: S307 — intentional restricted eval
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


@tool
def word_counter(text: str) -> str:
    """Count the number of words, characters, and sentences in a given text.

    Use this when the user asks about text statistics or word counts.
    """
    words = len(text.split())
    characters = len(text)
    sentences = len([s for s in text.split(".") if s.strip()])
    return (
        f"Text analysis:\n"
        f"  Words:      {words}\n"
        f"  Characters: {characters}\n"
        f"  Sentences:  {sentences}"
    )


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Returns temperature, conditions, and humidity.
    Use this when the user asks about weather in a specific location.
    Supported cities: London, New York, Tokyo, Sydney, Paris.
    """
    city_clean = city.splitlines()[0].strip().strip("'\"")
    mock_weather: dict[str, dict[str, str]] = {
        "london":   {"temp": "12°C", "condition": "Cloudy",       "humidity": "78%"},
        "new york": {"temp": "18°C", "condition": "Sunny",        "humidity": "55%"},
        "tokyo":    {"temp": "22°C", "condition": "Partly cloudy","humidity": "65%"},
        "sydney":   {"temp": "25°C", "condition": "Clear",        "humidity": "50%"},
        "paris":    {"temp": "15°C", "condition": "Rainy",        "humidity": "82%"},
    }
    key = city_clean.lower().strip()
    if key in mock_weather:
        w = mock_weather[key]
        return (
            f"Weather in {city_clean.title()}: "
            f"{w['temp']}, {w['condition']}, Humidity: {w['humidity']}"
        )
    return (
        f"Weather data not available for '{city_clean}'. "
        "Try: London, New York, Tokyo, Sydney, Paris."
    )


@tool
def text_summarizer(text: str) -> str:
    """Produce a brief 1-2 sentence summary of the provided text.

    Use this to condense long passages into key points.
    """
    words = text.split()
    if len(words) <= 20:
        return f"Text is already short: {text}"
    first_sentence = text.split(".")[0].strip()
    return f"Summary: {first_sentence}. (Original: {len(words)} words)"


# Convenience list — import this into agents
ALL_TOOLS = [calculator, word_counter, get_weather, text_summarizer]
