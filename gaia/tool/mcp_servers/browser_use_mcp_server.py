import asyncio
import logging
import sys
from fastmcp import FastMCP
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters  # (already imported in config.py)
# import your existing tool class
from tool.mcp_servers.auto_browser import *
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)
from tool.browser.logger import logger as browser_logger

mcp = FastMCP("browser-use-server")


@mcp.tool()
async def auto_browser_use(task: str) -> str:
    """Automate a browser to complete a high-level task.

    Args:
        task: Description of the overall task of what to do in the browser.

    Returns:
        Extracted content as a plain string.
    """
    if not task or not task.strip():
        return "[ERROR]: 'task' is required."

    try:
        model_id = "claude"

        api_key = os.getenv("GRADIENT_API_KEY")

        result = await browse(
            model_id=model_id,
            api_key=api_key,
            task=task,
        )
        return result

    except Exception as e:
        return f"[ERROR]: Exception {e} occurred in auto_browser_use."


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Browser Use MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="Transport method: 'stdio' or 'sse' (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to use when running with SSE transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8930,
        help="Port to use when running with SSE transport (default: 8930)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=args.host, port=args.port)
