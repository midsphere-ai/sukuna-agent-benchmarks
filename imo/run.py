"""Solve one IMO problem with a solver agent and a grader agent that reviews each attempt.

Usage:
    python run.py --q imo4
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from aworld.agents.llm_agent import Agent
from aworld.config.conf import AgentConfig, TaskConfig
from aworld.core.task import Task
from aworld.runner import Runners
from grader import Grader
from prompt import system_prompt

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("IMO_OUTPUT_DIR") or HERE / "output")

parser = argparse.ArgumentParser()
parser.add_argument("--q", type=str, required=True, help="Problem id from problems.jsonl, e.g. imo6.")
args = parser.parse_args()


def setup_logging():
    logging_logger = logging.getLogger()
    logging_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(OUTPUT_DIR / f"solution_{args.q}.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging_logger.addHandler(file_handler)


def load_problems(path: Path) -> List[Dict[str, Any]]:
    """Read problems.jsonl: one {"task_id", "Question"} object per line."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class GuardRunner:
    """Runs the solver, sends its output to the grader, and feeds the review back."""

    def __init__(self, super_agent: Agent, grader: Grader, original_task: str):
        self.super_agent = super_agent
        self.grader = grader
        self.original_task = original_task
        self.conversation_history = []
        self.max_iterations = 10  # max conversation rounds

    async def run_conversation(self, question: str) -> str:
        """run the conversation"""
        current_input = question
        iteration = 0

        while iteration < self.max_iterations:
            logging.info(f"=== The {iteration + 1} round of the conversation ===")

            # 1. Super-agent handles the current input
            logging.info(f"Super-agent input: {current_input[:100]}...")
            super_output = await self._call_super_agent(current_input)
            logging.info(f"Super-agent output: {super_output[:200]}...")

            # 2. Check if the output contains the final answer
            if self._has_final_answer(super_output) and iteration > 1:
                logging.info("find the final answer, the conversation should stop")
                return super_output

            if iteration == self.max_iterations - 1:
                logging.info("reach the max conversation rounds, the conversation should stop and return the final answer")
                return super_output

            # 3. Call the guard tool
            logging.info("guard tool is being called...")
            guard_output = await self._call_guard_tool(super_output)
            logging.info(f"guard tool output: {guard_output[:200]}...")

            # 4. Record the current round of the conversation history (before preparing the next input)
            self.conversation_history.append({
                "iteration": iteration + 1,
                "super_input": current_input,
                "super_output": super_output,
                "guard_output": guard_output
            })

            # 5. Prepare the next input
            current_input = self._prepare_next_input()

            iteration += 1

            # Check if the conversation should continue
            if self._should_stop_conversation(guard_output):
                logging.info("the conversation should stop")
                break

        logging.info(f"the converstation finished, there are totally {iteration} rounds")
        return super_output

    async def _call_super_agent(self, input_text: str) -> str:
        """call the super-agent"""
        try:
            task = Task(input=input_text, agent=self.super_agent, conf=TaskConfig())
            result = Runners.sync_run_task(task=task)
            if result and task.id in result:
                return result[task.id].answer
            return "Super-agent fail to return the result"
        except Exception as e:
            logging.error(f"fail to call the super-agent: {e}")
            return f"fail to call the super-agent: {str(e)}"

    async def _call_guard_tool(self, super_output: str) -> str:
        """call the guard tool"""
        try:
            return await self.grader.review(super_output, self.original_task)
        except Exception as e:
            logging.error(f"fail to call the guard tool: {e}")
            return f"fail to call the guard tool: {str(e)}"

    def _has_final_answer(self, output: str) -> bool:
        """check if the output contains the final answer"""
        answer_patterns = [
            r"<answer>.*?</answer>",
            r"Final answer[：:]\s*",
            r"The final answer is[：:]\s*",
            r"the final answer is\s*",
            r"I have successfully solved the problem."
        ]
        return any(re.search(pattern, output, re.IGNORECASE) for pattern in answer_patterns)

    def _prepare_next_input(self) -> str:
        """Prepare for the next round with complete conversation history"""
        conversation_text = f"Original Task:\n{self.original_task}\n\n"

        for history in self.conversation_history:
            conversation_text += f"=== Round {history['iteration']} ===\n"
            conversation_text += f"Previous solution:\n{history['super_output']}\n\n"
            conversation_text += f"IMO grader review:\n{history['guard_output']}\n\n"

        combined_input = f"""
{conversation_text}
Please seriously consider all the above reviews from the IMO grader across all rounds, and provide a refined and improved solution that addresses all the issues identified.
"""
        return combined_input.strip()

    def _should_stop_conversation(self, guard_output: str) -> bool:
        """check if the guard tool thinks the conversation should stop"""
        stop_indicators = [
            "The answer is completed",
            "No need to further refine",
            "The question has been correctly solved",
            "No loopholes or oversights found"
        ]
        return any(indicator in guard_output for indicator in stop_indicators)


if __name__ == "__main__":
    load_dotenv(HERE / ".env", override=True, verbose=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging()

    problems = load_problems(HERE / "problems.jsonl")
    logging.info(f"Total questions: {len(problems)}")

    # the solver has no tools
    agent_config = AgentConfig(
        llm_provider="openai",
        llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_temperature=0.1,
    )
    super_agent = Agent(
        conf=agent_config,
        name="imo_solver",
        system_prompt=system_prompt,
    )

    grader = Grader()

    # load results from the checkpoint file
    results_path = OUTPUT_DIR / "results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as results_f:
            results: List[Dict[str, Any]] = json.load(results_f)
    else:
        results: List[Dict[str, Any]] = []

    try:
        selected = [p for p in problems if p["task_id"] == args.q]
        if not selected:
            logging.error(f"Task ID '{args.q}' not found in dataset")
            sys.exit()

        for i, problem in enumerate(selected):
            try:
                logging.info(f"Start to process: {problem['task_id']}")
                logging.info(f"Question: {problem['Question']}")

                question = problem["Question"]
                guard_runner = GuardRunner(super_agent=super_agent, grader=grader, original_task=question)
                result = asyncio.run(guard_runner.run_conversation(question))

                new_result = {
                    "task_id": problem["task_id"],
                    "question": question,
                    "response": result,
                    "conversation_history": guard_runner.conversation_history,
                }

                # Replace the record for this task_id if there is one, otherwise append
                existing_index = next(
                    (j for j, r in enumerate(results) if r.get("task_id") == problem["task_id"]),
                    None,
                )
                if existing_index is not None:
                    results[existing_index] = new_result
                    logging.info(f"Updated existing record for task_id: {problem['task_id']}")
                else:
                    results.append(new_result)
                    logging.info(f"Added new record for task_id: {problem['task_id']}")

            except Exception:
                logging.error(f"Error processing {i}: {traceback.format_exc()}")
                continue
    except KeyboardInterrupt:
        pass
    finally:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
