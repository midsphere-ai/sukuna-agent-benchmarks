import logging
import os
import sys
import requests
import asyncio
import json
from typing import Optional
import urllib.parse
from markitdown import MarkItDown
import io
from dotenv import load_dotenv

logger = logging.getLogger("gaia-agent")
load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")


def request_to_json(content: str) -> dict:
    """Parse JSON content from a string, handling special prefixes from scrapers."""
    if isinstance(content, str) and "Markdown Content:\n" in content:
        # If the content starts with "Markdown Content:\n", extract only the part after it (from JINA)
        content = content.split("Markdown Content:\n")[1]
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse JSON: {str(e)}",
            e.doc,
            e.pos,
        ) from e


async def api_request_json(
    url: str,
    params: dict = None,
    timeout: int = 30,
    max_retries: int = 3,
    headers: dict = None,
) -> dict:
    """Make an API request that returns JSON with retry logic.

    Args:
        url: API endpoint URL.
        params: Query params.
        timeout: Request timeout in seconds.
        max_retries: Maximum attempts.
        headers: Optional headers (defaults to Wikipedia-friendly UA).
    """
    retry_count = 0
    last_error: Optional[Exception] = None

    if headers is None:
        headers = {
            "User-Agent": "gaia-agent/1.0"
        }

    while retry_count < max_retries:
        try:
            response = requests.get(url, params=params, timeout=timeout, headers=headers)
            response.raise_for_status()
            try:
                return response.json()
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    (
                        "API returned invalid JSON. "
                        f"Status: {response.status_code}, Preview: {response.text[:200]}"
                    ),
                    response.text,
                    e.pos,
                ) from e

        except requests.exceptions.Timeout as e:
            last_error = e
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(min(2**retry_count, 60))
            else:
                raise requests.exceptions.RequestException(
                    f"Request timed out after {max_retries} attempts: {str(e)}"
                ) from e

        except requests.exceptions.HTTPError as e:
            if e.response and 400 <= e.response.status_code < 500:
                raise requests.exceptions.RequestException(
                    f"HTTP {e.response.status_code} error: {str(e)}"
                ) from e
            last_error = e
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(min(2**retry_count, 60))
            else:
                raise requests.exceptions.RequestException(
                    f"HTTP error after {max_retries} attempts: {str(e)}"
                ) from e

        except requests.exceptions.RequestException as e:
            last_error = e
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(min(2**retry_count, 60))
            else:
                raise requests.exceptions.RequestException(
                    f"Request failed after {max_retries} attempts: {str(e)}"
                ) from e

    if last_error:
        raise last_error
    raise requests.exceptions.RequestException("Request failed for unknown reason")


async def smart_request(url: str, params: dict = None) -> str:
    # Handle empty URL
    if not url:
        return f"[ERROR]: Invalid URL: '{url}'. URL cannot be empty."

    # Auto-add https:// if no protocol is specified
    protocol_hint = ""
    if not url.startswith(("http://", "https://")):
        original_url = url
        url = f"https://{url}"
        protocol_hint = f"[NOTE]: Automatically added 'https://' to URL '{original_url}' -> '{url}'\n\n"

    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    # Check for restricted domains
    if "huggingface.co/datasets" in url or "huggingface.co/spaces" in url:
        return "You are trying to scrape a Hugging Face dataset for answers, please do not use the scrape tool for this purpose."

    if "arxiv.org/src" in url:
        return "You are currently scraping the source files (LaTeX packages) of arXiv papers, which are not useful for solving the task. Please modify your request content."

    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            error_msg = "[NOTE]: If the link is a file / image / video / audio, please use other applicable tools, or try to process it in the sandbox.\n"
            youtube_hint = ""
            if (
                "youtube.com/watch" in url
                or "youtube.com/shorts" in url
                or "youtube.com/live" in url
            ):
                youtube_hint = "[NOTE]: If you need to get information about its visual or audio content, please use tool 'visual_audio_youtube_analyzing' instead. This tool may not be able to provide visual and audio content of a YouTube Video.\n\n"


            ### Then Jina.ai
            content, jina_err = await scrape_jina(url)
            if jina_err:
                error_msg += (
                    f"[ERROR]: Failed to get content from Jina.ai: {jina_err}\n"
                )
            elif content is None or content.strip() == "":
                error_msg += "[ERROR]: No content got from Jina.ai.\n"
            else:
                logger.info("Successfully got content from Jina.ai")
                return protocol_hint + youtube_hint + content

            content, request_err = scrape_request(url)
            if request_err:
                error_msg += (
                    f"[ERROR]: Failed to get content from requests: {request_err}\n"
                )
            elif content is None or content.strip() == "":
                error_msg += "[ERROR]: No content got from requests.\n"
            else:
                logger.info("Successfully got content from requests via scrape_request")
                return protocol_hint + youtube_hint + content

            raise Exception(error_msg)

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                return str(e)
            else:
                await asyncio.sleep(4**retry_count)


async def scrape_jina(url: str) -> tuple[str, str]:
    # Use Jina.ai reader API to convert URL to LLM-friendly text
    if JINA_API_KEY == "":
        return (
            None,
            "[ERROR]: JINA_API_KEY is not set, scrape_website tool is not available.",
        )

    jina_headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Base": "final",
        "X-Engine": "browser",
        "X-With-Generated-Alt": "true",
        "X-With-Iframe": "true",
        "X-With-Shadow-Dom": "true",
    }

    jina_url = f"https://r.jina.ai/{url}"
    try:
        response = requests.get(jina_url, headers=jina_headers, timeout=120)
        if response.status_code == 422:
            # Return as error to allow fallback to other tools and retries
            return (
                None,
                "[ERROR]: Tool execution failed with 422 error, which may indicate the URL is a file. This tool does not support files. If you believe the URL might point to a file, you should try using other applicable tools, or try to process it in the sandbox.",
            )
        response.raise_for_status()
        content = response.text
        if (
            "Warning: This page maybe not yet fully loaded, consider explicitly specify a timeout."
            in content
        ):
            # Try with longer timeout
            response = requests.get(jina_url, headers=jina_headers, timeout=300)
            if response.status_code == 422:
                return (
                    None,
                    "[ERROR]: Tool execution failed with 422 error, which may indicate the URL is a file. This tool does not support files. If you believe the URL might point to a file, you should try using other applicable tools, or try to process it in the sandbox.",
                )
            response.raise_for_status()
            content = response.text
        return content, None
    except Exception as e:
        return None, f"[ERROR]: Failed to get content from Jina.ai: {str(e)}\n"


def scrape_request(url: str) -> tuple[str, str]:
    """This function uses requests to scrape a website.
    Args:
        url: The URL of the website to scrape.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        try:
            stream = io.BytesIO(response.content)
            md = MarkItDown()
            content = md.convert_stream(stream).text_content
            return content, None
        except Exception:
            # If MarkItDown conversion fails, return raw response text
            return response.text, None

    except Exception as e:
        return None, f"[ERROR]: {str(e)}"
