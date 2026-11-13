import asyncio
import os
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv(verbose=True)

from fastmcp import FastMCP
from openai import AsyncOpenAI

BOUNDARY = 7
GRADIENT_API_KEY = os.getenv("GRADIENT_API_KEY", "")
GRADIENT_BASE_URL = os.getenv("GRADIENT_BASE_URL", "https://inference.do-ai.run/v1")
DOUBTER_MODEL = "openai-gpt-5.5"

INSTRUCTION = dedent(
    """\
You should first analyze this flow of actions taken by the agent.
As you are a naturally doubtful tool, try to spot out places within the flow of actions where there might be a logical flow or a missing step.
If there are any flaws within the flow of action, I need you to tell me that I need to run the agent again and suggest me an improvement I can make to the flow of actions to be taken by the agent.
Not all tasks will have flaws, some might be perfect in which case you should suggest no improvements.
But do keep in mind that the rest might not be, and you need to discern out these flawed sequence of actions.
The history shown to you is all the action taken so far right before the verification step which is going to happen now. The key mistake in the code is not that the verification step hasn't run yet even though the previous steps show that the verification step haven't run yet. That's because you are going to run now. SO DONT POINT OUT THAT AS AN ERROR.

Here is the history of the flow of actions:\n
"""
)

SCORE_INSTRUCTION = dedent(
    f"""\
You will be given a reflection of the past actions taken by the agent by another LLM.
I need you to look at the reflection done by this LLM and give an overall score from 0 to 10.
The score reflects the past actions taken by the original agent. If you give it a score of 0, it means the past actions taken were illogical and full of errors and flaws, requiring a reset of the plans and another run by the agent.
A score of 10 means the past actions were perfect, logically sound and robust enough that the agent does not need to perform it again.
After deciding on the score, give me a decision whether to run the agent again one more time. The boundary score of whether to rerun is {BOUNDARY}.
Any score lower than this requires the agent to run again.
Any score higher or equal to this does not require the agent to run again as it means the past actions were logically sound enough.

Give me a score, the respective decision, and the reasoning behind that decision, after considering the reflection of the past actions shown below:\n
"""
)

mcp = FastMCP("doubter-mcp-server")

_gradient_client = AsyncOpenAI(
    api_key=GRADIENT_API_KEY,
    base_url=GRADIENT_BASE_URL,
    timeout=60,
)


async def _call_model(prompt: str, max_retries: int = 3) -> str:
    if not GRADIENT_API_KEY:
        raise RuntimeError("GRADIENT_API_KEY is not configured")

    for attempt in range(1, max_retries + 1):
        try:
            response = await _gradient_client.chat.completions.create(
                model=DOUBTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            if not response or not response.choices:
                raise RuntimeError("Empty response from doubter model")
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Response contained no message content")
            return content
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Doubter call failed after {attempt} attempts: {exc}"
                ) from exc
            await asyncio.sleep(min(2 ** attempt, 8))


@mcp.tool()
async def doubter(history: str) -> str:
    """Critically review the agent's action history and decide if a rerun is needed."""
    if not history or not history.strip():
        return "[ERROR]: 'history' is required."

    history = history.strip()

    try:
        critique = await _call_model(INSTRUCTION + history)
        score_and_decision = await _call_model(SCORE_INSTRUCTION + critique)
    except RuntimeError as exc:
        return f"[ERROR]: {exc}"

    return (
        "The history of the agent's actions:\n"
        f"{history}\n\n"
        "Doubter review:\n"
        f"{critique}\n\n"
        "Score & decision:\n"
        f"{score_and_decision}\n"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
