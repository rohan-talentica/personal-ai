"""
Notion integration tool — fetch daily learning notes from a Notion database.

The Notion database is expected to have:
  - A **Name/Title** property (the page title)
  - A **Date** property (type: Date) indicating when the notes were written

Environment variables required:
  NOTION_API_KEY        — Notion internal integration secret
  NOTION_DATABASE_ID    — ID of the database to query
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Optional

from notion_client import Client
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Notion client (lazy singleton) ───────────────────────────────────────────

_client: Optional[Client] = None


def _get_client() -> Client:
    global _client
    if _client is None:
        api_key = os.getenv("NOTION_API_KEY")
        if not api_key:
            raise ValueError("NOTION_API_KEY environment variable is not set.")
        _client = Client(auth=api_key)
    return _client


def _get_database_id() -> str:
    db_id = os.getenv("NOTION_DATABASE_ID")
    if not db_id:
        raise ValueError("NOTION_DATABASE_ID environment variable is not set.")
    return db_id


# ── Core functions ────────────────────────────────────────────────────────────

def fetch_pages_by_date(date_str: str) -> list[dict]:
    """Query the Notion database for pages whose Date property matches date_str.

    Args:
        date_str: ISO date string, e.g. "2026-03-03"

    Returns:
        List of raw Notion page objects.
    """
    client = _get_client()
    database_id = _get_database_id()

    logger.info("Querying Notion DB for date: %s", date_str)

    try:
        response = client.databases.query(
            database_id=database_id,
            filter={
                "property": "Date",
                "date": {
                    "equals": date_str
                }
            }
        )
        pages = response.get("results", [])
        logger.info("Found %d page(s) for %s", len(pages), date_str)
        return pages
    except Exception as exc:
        logger.error("Notion database query failed: %s", exc)
        raise


def extract_page_content(page_id: str) -> str:
    """Fetch all block-level text content from a Notion page.

    Concatenates all paragraph, heading, bulleted_list_item,
    numbered_list_item, toggle, quote, and callout blocks.

    Args:
        page_id: The Notion page UUID.

    Returns:
        Plain text string of the page content.
    """
    client = _get_client()
    text_parts: list[str] = []

    try:
        # Pagination — Notion returns at most 100 blocks per request
        cursor = None
        while True:
            kwargs: dict = {"block_id": page_id, "page_size": 100}
            if cursor:
                kwargs["start_cursor"] = cursor

            response = client.blocks.children.list(**kwargs)
            blocks = response.get("results", [])

            for block in blocks:
                block_type = block.get("type", "")
                block_data = block.get(block_type, {})
                rich_text = block_data.get("rich_text", [])

                # Flatten rich_text array → plain string
                text = "".join(rt.get("plain_text", "") for rt in rich_text)
                if text.strip():
                    # Add a prefix for headings to preserve structure
                    if block_type == "heading_1":
                        text_parts.append(f"\n# {text}")
                    elif block_type == "heading_2":
                        text_parts.append(f"\n## {text}")
                    elif block_type == "heading_3":
                        text_parts.append(f"\n### {text}")
                    elif block_type == "bulleted_list_item":
                        text_parts.append(f"• {text}")
                    elif block_type == "numbered_list_item":
                        text_parts.append(f"- {text}")
                    else:
                        text_parts.append(text)

            if not response.get("has_more"):
                break
            cursor = response.get("next_cursor")

    except Exception as exc:
        logger.error("Failed to fetch blocks for page %s: %s", page_id, exc)
        raise

    return "\n".join(text_parts)


def get_page_title(page: dict) -> str:
    """Extract the title from a raw Notion page object."""
    try:
        # Title is stored in the first title-type property
        props = page.get("properties", {})
        for prop in props.values():
            if prop.get("type") == "title":
                rich_text = prop.get("title", [])
                return "".join(rt.get("plain_text", "") for rt in rich_text)
    except Exception:
        pass
    return "Untitled"


def get_daily_notes(date_str: str) -> list[dict]:
    """Fetch all Notion pages for the given date and their full text content.

    Args:
        date_str: ISO date string, e.g. "2026-03-03"

    Returns:
        List of dicts: [{"title": ..., "content": ..., "date": ..., "page_id": ...}]
    """
    pages = fetch_pages_by_date(date_str)

    notes = []
    for page in pages:
        page_id = page["id"]
        title = get_page_title(page)

        logger.info("Fetching content for page: %s (%s)", title, page_id)
        content = extract_page_content(page_id)

        notes.append({
            "title": title,
            "content": content,
            "date": date_str,
            "page_id": page_id,
        })

    return notes
