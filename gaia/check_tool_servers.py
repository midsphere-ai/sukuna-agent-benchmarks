"""
Comprehensive test for all GAIA MCP server tools.
Run from gaia/ with: tool/.venv-tool/bin/python check_tool_servers.py
"""
import asyncio
import os
import sys
import time
import tempfile
import traceback
from dotenv import load_dotenv

load_dotenv(verbose=True)

# Track results: key -> (status, detail, elapsed_seconds)
results = {}


def unwrap(tool):
    """Unwrap a FastMCP FunctionTool to get the actual async function."""
    return getattr(tool, "fn", tool)


def report(server: str, tool_name: str, status: str, detail: str = "", elapsed: float = 0.0):
    key = f"{server}::{tool_name}"
    results[key] = (status, detail, elapsed)
    icon = "PASS" if status == "PASS" else "FAIL" if status == "FAIL" else "SKIP"
    print(f"  [{icon}] {tool_name} ({elapsed:.1f}s): {detail[:200]}")


async def timed(coro):
    """Run a coroutine and return (result, elapsed_seconds)."""
    t0 = time.monotonic()
    result = await coro
    return result, time.monotonic() - t0


async def test_reading_server():
    """Test read_file tool from reading_mcp_server."""
    print("\n" + "=" * 60)
    print("1. READING MCP SERVER (reading_mcp_server.py)")
    print("=" * 60)

    from tool.mcp_servers.reading_mcp_server import read_file as _read_file
    read_file = unwrap(_read_file)

    # Test 1a: Read a local text file
    print("\n--- Test 1a: Read local text file ---")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello from test file!\nLine 2\nLine 3")
            tmp_path = f.name
        result, elapsed = await timed(read_file(tmp_path))
        os.unlink(tmp_path)
        if "Hello from test file" in result:
            report("reading", "read_file(local_txt)", "PASS", f"Content: {result[:100]}", elapsed)
        else:
            report("reading", "read_file(local_txt)", "FAIL", f"Unexpected: {result[:200]}", elapsed)
    except Exception as e:
        report("reading", "read_file(local_txt)", "FAIL", f"Exception: {e}")

    # Test 1b: Read a local CSV file
    print("\n--- Test 1b: Read local CSV file ---")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
            tmp_csv = f.name
        result, elapsed = await timed(read_file(tmp_csv))
        os.unlink(tmp_csv)
        if "Alice" in result or "name" in result:
            report("reading", "read_file(local_csv)", "PASS", f"Content: {result[:100]}", elapsed)
        else:
            report("reading", "read_file(local_csv)", "FAIL", f"Unexpected: {result[:200]}", elapsed)
    except Exception as e:
        report("reading", "read_file(local_csv)", "FAIL", f"Exception: {e}")

    # Test 1c: Read a remote URL
    print("\n--- Test 1c: Read remote URL ---")
    try:
        result, elapsed = await timed(read_file("https://httpbin.org/robots.txt"))
        if result and len(result) > 5 and "[ERROR]" not in result:
            report("reading", "read_file(remote_url)", "PASS", f"Got {len(result)} chars: {result[:100]}", elapsed)
        else:
            report("reading", "read_file(remote_url)", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("reading", "read_file(remote_url)", "FAIL", f"Exception: {e}")

    # Test 1d: Invalid URI
    print("\n--- Test 1d: Invalid URI ---")
    try:
        result, elapsed = await timed(read_file(""))
        if "[ERROR]" in result:
            report("reading", "read_file(empty_uri)", "PASS", f"Correctly rejected: {result[:100]}", elapsed)
        else:
            report("reading", "read_file(empty_uri)", "FAIL", f"Should have errored: {result[:200]}", elapsed)
    except Exception as e:
        report("reading", "read_file(empty_uri)", "FAIL", f"Exception: {e}")

    # Test 1e: Sandbox path rejection
    print("\n--- Test 1e: Sandbox path rejection ---")
    try:
        result, elapsed = await timed(read_file("/home/user/something.txt"))
        if "sandbox" in result.lower() or "cannot access" in result.lower():
            report("reading", "read_file(sandbox_path)", "PASS", f"Correctly rejected: {result[:100]}", elapsed)
        else:
            report("reading", "read_file(sandbox_path)", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("reading", "read_file(sandbox_path)", "FAIL", f"Exception: {e}")


async def test_searching_server():
    """Test general_search tool from searching_mcp_server."""
    print("\n" + "=" * 60)
    print("2. SEARCHING MCP SERVER (searching_mcp_server.py)")
    print("=" * 60)

    from tool.mcp_servers.searching_mcp_server import general_search as _gs, google_search
    general_search = unwrap(_gs)

    # Test 2a: Complex multi-step research query
    print("\n--- Test 2a: General search (complex research query) ---")
    try:
        result, elapsed = await timed(general_search(
            "What were the key differences between the Treaty of Westphalia 1648 and the Congress of Vienna 1815 in terms of how they reshaped European borders and sovereignty?",
            mode="balanced", engine_timeout=180,
        ))
        if result and "[ERROR]" not in result and len(result) > 50:
            report("searching", "general_search(complex)", "PASS", f"Got {len(result)} chars: {result[:150]}", elapsed)
        else:
            report("searching", "general_search(complex)", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("searching", "general_search(complex)", "FAIL", f"Exception: {e}")

    # Test 2b: Empty query
    print("\n--- Test 2b: General search (empty query) ---")
    try:
        result, elapsed = await timed(general_search(""))
        if "[ERROR]" in result:
            report("searching", "general_search(empty)", "PASS", f"Correctly rejected: {result[:100]}", elapsed)
        else:
            report("searching", "general_search(empty)", "FAIL", f"Should have errored: {result[:200]}", elapsed)
    except Exception as e:
        report("searching", "general_search(empty)", "FAIL", f"Exception: {e}")

    # Test 2c: Google search directly
    print("\n--- Test 2c: Google search (Serper) ---")
    try:
        result, elapsed = await timed(google_search("site:arxiv.org transformer attention mechanism 2024"))
        if result and "[ERROR]" not in result and len(result) > 20:
            report("searching", "google_search", "PASS", f"Got {len(result)} chars: {result[:150]}", elapsed)
        else:
            report("searching", "google_search", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("searching", "google_search", "FAIL", f"Exception: {e}")


async def test_python_server():
    """Test E2B sandbox tools from python_server."""
    print("\n" + "=" * 60)
    print("3. PYTHON/E2B SERVER (python_server.py)")
    print("=" * 60)

    from tool.mcp_servers.python_server import (
        create_sandbox as _cs, run_command as _rc, run_python_code as _rpc,
        upload_file_from_local_to_sandbox as _uf,
        download_file_from_internet_to_sandbox as _di,
        download_file_from_sandbox_to_local as _ds,
    )
    create_sandbox = unwrap(_cs)
    run_command = unwrap(_rc)
    run_python_code = unwrap(_rpc)
    upload_file_from_local_to_sandbox = unwrap(_uf)
    download_file_from_internet_to_sandbox = unwrap(_di)
    download_file_from_sandbox_to_local = unwrap(_ds)

    sandbox_id = None

    # Test 3a: Create sandbox
    print("\n--- Test 3a: Create sandbox ---")
    try:
        result, elapsed = await timed(create_sandbox())
        if "sandbox_id" in result.lower() and "[ERROR]" not in result:
            sandbox_id = result.split("sandbox_id: ")[-1].strip()
            report("python", "create_sandbox", "PASS", f"Created: {sandbox_id}", elapsed)
        else:
            report("python", "create_sandbox", "FAIL", f"Result: {result[:200]}", elapsed)
            return
    except Exception as e:
        report("python", "create_sandbox", "FAIL", f"Exception: {e}")
        return

    # Test 3b: Run a shell command
    print("\n--- Test 3b: Run shell command ---")
    try:
        result, elapsed = await timed(run_command("echo 'Hello E2B!' && uname -a", sandbox_id))
        if "Hello E2B" in result:
            report("python", "run_command", "PASS", f"Output: {result[:150]}", elapsed)
        else:
            report("python", "run_command", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("python", "run_command", "FAIL", f"Exception: {e}")

    # Test 3c: Run Python code
    print("\n--- Test 3c: Run Python code ---")
    try:
        code = "import math; print(f'Pi = {math.pi:.10f}'); print(f'sqrt(2) = {math.sqrt(2):.10f}')"
        result, elapsed = await timed(run_python_code(code, sandbox_id))
        if "3.14159" in result:
            report("python", "run_python_code", "PASS", f"Output: {result[:150]}", elapsed)
        else:
            report("python", "run_python_code", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("python", "run_python_code", "FAIL", f"Exception: {e}")

    # Test 3d: Upload local file to sandbox
    print("\n--- Test 3d: Upload file to sandbox ---")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test upload content\nLine 2\n")
            tmp_upload = f.name
        result, elapsed = await timed(upload_file_from_local_to_sandbox(sandbox_id, tmp_upload))
        os.unlink(tmp_upload)
        if "uploaded" in result.lower() and "[ERROR]" not in result:
            report("python", "upload_file_to_sandbox", "PASS", f"Result: {result[:150]}", elapsed)
        else:
            report("python", "upload_file_to_sandbox", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("python", "upload_file_to_sandbox", "FAIL", f"Exception: {e}")

    # Test 3e: Download file from internet to sandbox
    print("\n--- Test 3e: Download file from internet to sandbox ---")
    try:
        result, elapsed = await timed(download_file_from_internet_to_sandbox(
            sandbox_id, "https://httpbin.org/robots.txt"
        ))
        if "downloaded" in result.lower() and "[ERROR]" not in result:
            report("python", "download_internet_to_sandbox", "PASS", f"Result: {result[:150]}", elapsed)
        else:
            report("python", "download_internet_to_sandbox", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("python", "download_internet_to_sandbox", "FAIL", f"Exception: {e}")

    # Test 3f: Download file from sandbox to local
    print("\n--- Test 3f: Download file from sandbox to local ---")
    try:
        await run_command("echo 'sandbox file content' > /home/user/test_download.txt", sandbox_id)
        result, elapsed = await timed(download_file_from_sandbox_to_local(sandbox_id, "/home/user/test_download.txt"))
        if "downloaded" in result.lower() and "[ERROR]" not in result:
            report("python", "download_sandbox_to_local", "PASS", f"Result: {result[:150]}", elapsed)
        else:
            report("python", "download_sandbox_to_local", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("python", "download_sandbox_to_local", "FAIL", f"Exception: {e}")

    # Test 3g: Invalid sandbox_id
    print("\n--- Test 3g: Invalid sandbox ID ---")
    try:
        result, elapsed = await timed(run_command("echo test", "invalid-sandbox-id-12345"))
        if "[ERROR]" in result:
            report("python", "run_command(bad_id)", "PASS", f"Correctly rejected: {result[:100]}", elapsed)
        else:
            report("python", "run_command(bad_id)", "FAIL", f"Should have errored: {result[:200]}", elapsed)
    except Exception as e:
        report("python", "run_command(bad_id)", "FAIL", f"Exception: {e}")


async def test_vision_server():
    """Test visual_question_answering and youtube tools from vision_mcp_server."""
    print("\n" + "=" * 60)
    print("4. VISION MCP SERVER (vision_mcp_server.py)")
    print("=" * 60)

    from tool.mcp_servers.vision_mcp_server import (
        visual_question_answering as _vqa,
        visual_audio_youtube_analyzing as _yt,
    )
    visual_question_answering = unwrap(_vqa)
    visual_audio_youtube_analyzing = unwrap(_yt)

    # Test 4a: Visual QA on a public image URL (Google-hosted, no rate limits)
    print("\n--- Test 4a: Visual QA on a public image URL ---")
    try:
        result, elapsed = await timed(visual_question_answering(
            "https://storage.googleapis.com/generativeai-downloads/images/scones.jpg",
            "What food is shown in this image?"
        ))
        if result and "[ERROR]" not in result and "failed" not in result.lower() and len(result) > 20:
            report("vision", "visual_qa(url_image)", "PASS", f"Got {len(result)} chars: {result[:150]}", elapsed)
        else:
            report("vision", "visual_qa(url_image)", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("vision", "visual_qa(url_image)", "FAIL", f"Exception: {e}")

    # Test 4b: Visual QA on a local image (pre-downloaded JPEG)
    print("\n--- Test 4b: Visual QA on local image ---")
    try:
        local_img = os.path.join(os.path.dirname(__file__), "test_image.jpg")
        result, elapsed = await timed(visual_question_answering(local_img, "What food is shown in this image?"))
        if result and "[ERROR]" not in result and "failed" not in result.lower() and len(result) > 20:
            report("vision", "visual_qa(local_image)", "PASS", f"Got {len(result)} chars: {result[:150]}", elapsed)
        else:
            report("vision", "visual_qa(local_image)", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("vision", "visual_qa(local_image)", "FAIL", f"Exception: {e}")

    # Test 4c: Sandbox path rejection
    print("\n--- Test 4c: Sandbox path rejection ---")
    try:
        result, elapsed = await timed(visual_question_answering("/home/user/image.png", "What is this?"))
        if "sandbox" in result.lower() or "cannot access" in result.lower():
            report("vision", "visual_qa(sandbox_path)", "PASS", f"Correctly rejected: {result[:100]}", elapsed)
        else:
            report("vision", "visual_qa(sandbox_path)", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("vision", "visual_qa(sandbox_path)", "FAIL", f"Exception: {e}")

    # Test 4d: YouTube video analysis
    print("\n--- Test 4d: YouTube video QA ---")
    try:
        result, elapsed = await timed(visual_audio_youtube_analyzing(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            question="What is this video about? Describe briefly."
        ))
        if result and "[ERROR]" not in result and len(result) > 20:
            report("vision", "youtube_analyzing(question)", "PASS", f"Got {len(result)} chars: {result[:150]}", elapsed)
        else:
            report("vision", "youtube_analyzing(question)", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("vision", "youtube_analyzing(question)", "FAIL", f"Exception: {e}")

    # Test 4e: YouTube invalid URL
    print("\n--- Test 4e: YouTube invalid URL ---")
    try:
        result, elapsed = await timed(visual_audio_youtube_analyzing("https://example.com/not-youtube", question="test"))
        if "[ERROR]" in result:
            report("vision", "youtube_analyzing(bad_url)", "PASS", f"Correctly rejected: {result[:100]}", elapsed)
        else:
            report("vision", "youtube_analyzing(bad_url)", "FAIL", f"Should have errored: {result[:200]}", elapsed)
    except Exception as e:
        report("vision", "youtube_analyzing(bad_url)", "FAIL", f"Exception: {e}")


async def test_audio_server():
    """Test audio tools from audio_mcp_server."""
    print("\n" + "=" * 60)
    print("5. AUDIO MCP SERVER (audio_mcp_server.py)")
    print("=" * 60)

    from tool.mcp_servers.audio_mcp_server import (
        audio_transcription as _at,
        audio_question_answering as _aqa,
        audio_metadata as _am,
    )
    audio_transcription = unwrap(_at)
    audio_question_answering = unwrap(_aqa)
    audio_metadata = unwrap(_am)

    # Download a small public audio clip for local tests
    test_audio_url = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
    print("\n--- Preparing: Downloading test audio (first 100KB) ---")
    local_audio_path = None
    try:
        import requests
        resp = requests.get(test_audio_url, stream=True, timeout=15)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            bytes_written = 0
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bytes_written += len(chunk)
                if bytes_written >= 100_000:
                    break
            local_audio_path = f.name
        print(f"  Downloaded {bytes_written} bytes to {local_audio_path}")
    except Exception as e:
        print(f"  Failed to download test audio: {e}")

    # Test 5a: Audio transcription (local file)
    print("\n--- Test 5a: Audio transcription (local) ---")
    if local_audio_path:
        try:
            result, elapsed = await timed(audio_transcription(local_audio_path))
            if result and "[ERROR]" not in result and len(result) > 5:
                report("audio", "audio_transcription(local)", "PASS", f"Transcription: {result[:150]}", elapsed)
            else:
                report("audio", "audio_transcription(local)", "FAIL", f"Result: {result[:200]}", elapsed)
        except Exception as e:
            report("audio", "audio_transcription(local)", "FAIL", f"Exception: {e}")
    else:
        report("audio", "audio_transcription(local)", "SKIP", "No local audio file available")

    # Test 5b: Audio QA (local file)
    print("\n--- Test 5b: Audio question answering (local) ---")
    if local_audio_path:
        try:
            result, elapsed = await timed(audio_question_answering(
                local_audio_path,
                "What type of music is this? Is there singing or just instruments?"
            ))
            if result and "[ERROR]" not in result and len(result) > 10:
                report("audio", "audio_qa(local)", "PASS", f"Answer: {result[:150]}", elapsed)
            else:
                report("audio", "audio_qa(local)", "FAIL", f"Result: {result[:200]}", elapsed)
        except Exception as e:
            report("audio", "audio_qa(local)", "FAIL", f"Exception: {e}")
    else:
        report("audio", "audio_qa(local)", "SKIP", "No local audio file available")

    # Test 5c: Audio metadata (local file - ACR_ACCESS_KEY may be missing)
    print("\n--- Test 5c: Audio metadata (local) ---")
    if local_audio_path:
        try:
            result, elapsed = await timed(audio_metadata(local_audio_path))
            if result and "[ERROR]" not in result:
                report("audio", "audio_metadata(local)", "PASS", f"Metadata: {result[:150]}", elapsed)
            else:
                report("audio", "audio_metadata(local)", "FAIL", f"Result: {result[:200]}", elapsed)
        except Exception as e:
            report("audio", "audio_metadata(local)", "FAIL", f"Exception: {e}")
    else:
        report("audio", "audio_metadata(local)", "SKIP", "No local audio file available")

    # Test 5d: Sandbox path rejection
    print("\n--- Test 5d: Sandbox path rejection ---")
    try:
        result, elapsed = await timed(audio_transcription("/home/user/audio.mp3"))
        if "sandbox" in result.lower() or "cannot access" in result.lower():
            report("audio", "audio_transcription(sandbox)", "PASS", f"Correctly rejected: {result[:100]}", elapsed)
        else:
            report("audio", "audio_transcription(sandbox)", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("audio", "audio_transcription(sandbox)", "FAIL", f"Exception: {e}")

    # Cleanup
    if local_audio_path and os.path.exists(local_audio_path):
        os.unlink(local_audio_path)


async def test_reasoning_server():
    """Test reasoning tool from reasoning_mcp_server."""
    print("\n" + "=" * 60)
    print("6. REASONING MCP SERVER (reasoning_mcp_server.py)")
    print("=" * 60)

    from tool.mcp_servers.reasoning_mcp_server import reasoning as _r
    reasoning = unwrap(_r)

    # Test 6a: Math problem
    print("\n--- Test 6a: Math reasoning ---")
    try:
        result, elapsed = await timed(reasoning("What is 17 * 23 + 45 - 12? Show your work step by step."))
        if result and "[ERROR]" not in result and len(result) > 10:
            report("reasoning", "reasoning(math)", "PASS", f"Answer: {result[:150]}", elapsed)
        else:
            report("reasoning", "reasoning(math)", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("reasoning", "reasoning(math)", "FAIL", f"Exception: {e}")

    # Test 6b: Logic puzzle
    print("\n--- Test 6b: Logic puzzle ---")
    try:
        result, elapsed = await timed(reasoning(
            "If all roses are flowers, and some flowers fade quickly, "
            "can we conclude that some roses fade quickly? Explain."
        ))
        if result and "[ERROR]" not in result and len(result) > 20:
            report("reasoning", "reasoning(logic)", "PASS", f"Answer: {result[:150]}", elapsed)
        else:
            report("reasoning", "reasoning(logic)", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("reasoning", "reasoning(logic)", "FAIL", f"Exception: {e}")

    # Test 6c: Empty question
    print("\n--- Test 6c: Empty question ---")
    try:
        result, elapsed = await timed(reasoning(""))
        if "[ERROR]" in result:
            report("reasoning", "reasoning(empty)", "PASS", f"Correctly rejected: {result[:100]}", elapsed)
        else:
            report("reasoning", "reasoning(empty)", "FAIL", f"Should have errored: {result[:200]}", elapsed)
    except Exception as e:
        report("reasoning", "reasoning(empty)", "FAIL", f"Exception: {e}")


async def test_browser_server():
    """Test auto_browser_use tool from browser_use_mcp_server."""
    print("\n" + "=" * 60)
    print("7. BROWSER USE MCP SERVER (browser_use_mcp_server.py)")
    print("=" * 60)

    from tool.mcp_servers.browser_use_mcp_server import auto_browser_use as _abu
    auto_browser_use = unwrap(_abu)

    # Test 7a: Simple browser task
    print("\n--- Test 7a: Browser - fetch info from a website ---")
    try:
        result, elapsed = await timed(auto_browser_use(
            "Go to https://en.wikipedia.org/wiki/Python_(programming_language) and tell me the year Python was first released."
        ))
        if result and "[ERROR]" not in result and len(result) > 10:
            report("browser", "auto_browser_use(wiki)", "PASS", f"Got {len(result)} chars: {result[:200]}", elapsed)
        else:
            report("browser", "auto_browser_use(wiki)", "FAIL", f"Result: {result[:200]}", elapsed)
    except Exception as e:
        report("browser", "auto_browser_use(wiki)", "FAIL", f"Exception: {e}")

    # Test 7b: Empty task rejection
    print("\n--- Test 7b: Browser - empty task ---")
    try:
        result, elapsed = await timed(auto_browser_use(""))
        if "[ERROR]" in result:
            report("browser", "auto_browser_use(empty)", "PASS", f"Correctly rejected: {result[:100]}", elapsed)
        else:
            report("browser", "auto_browser_use(empty)", "FAIL", f"Should have errored: {result[:200]}", elapsed)
    except Exception as e:
        report("browser", "auto_browser_use(empty)", "FAIL", f"Exception: {e}")


async def main():
    print("=" * 60)
    print("  GAIA MCP SERVER COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    start = time.time()

    await test_reading_server()
    await test_searching_server()
    await test_python_server()
    await test_vision_server()
    await test_audio_server()
    await test_reasoning_server()
    await test_browser_server()

    elapsed = time.time() - start

    # Print summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    pass_count = sum(1 for s, _, _ in results.values() if s == "PASS")
    fail_count = sum(1 for s, _, _ in results.values() if s == "FAIL")
    skip_count = sum(1 for s, _, _ in results.values() if s == "SKIP")

    # Header
    print(f"  {'Test':<45} {'Status':<6} {'Time':>7}")
    print(f"  {'-'*45} {'-'*6} {'-'*7}")
    for key, (status, detail, t) in results.items():
        icon = {"PASS": "OK", "FAIL": "XX", "SKIP": "--"}[status]
        print(f"  [{icon}] {key:<42} {t:>6.1f}s")
        if status == "FAIL":
            print(f"        -> {detail[:200]}")

    print(f"\nTotal: {len(results)} tests | PASS: {pass_count} | FAIL: {fail_count} | SKIP: {skip_count}")
    print(f"Total time: {elapsed:.1f}s")
    return fail_count


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(min(exit_code, 1))
