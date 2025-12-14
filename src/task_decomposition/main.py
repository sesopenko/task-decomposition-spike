import logging
from datetime import datetime
from pathlib import Path

from pydantic_ai import Agent

from task_decomposition.cost_calculator import calculate_cost
from task_decomposition.models_schema import TaskPlan
from task_decomposition.task_plan_builder import DefaultTaskPlanAgentBuilder
from task_decomposition.task_plan_validator import TaskPlanValidator
from task_decomposition.task_plan_executor import TaskPlanExecutor
from task_decomposition.delegate_runner import DelegateRunner, set_run_output_dir, OUTPUT_ROOT
import inflect

p = inflect.engine()
logger = logging.getLogger(__name__)


def _initialise_run_output_dir() -> Path:
    """
    Create and return a per-run output directory under the project-level
    'output' directory.

    The directory name is based on the current local date and time to keep
    outputs from different runs isolated, for example:

        output/2025-03-01_14-23-45
    """
    # Ensure the root output directory exists
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run output directory initialised at: %s", run_dir)
    return run_dir


def main():
    # Basic logging configuration; adjust as needed by the application
    logging.basicConfig(level=logging.INFO)

    # Initialise per-run output directory and configure the save_file tool
    run_output_dir = _initialise_run_output_dir()
    set_run_output_dir(run_output_dir)

    # Build the TaskPlan-producing agent via the abstraction
    builder = DefaultTaskPlanAgentBuilder()
    agent: Agent[TaskPlan] = builder.build_agent()

    logger.info("Using LLM to get task plan...")

    validator = TaskPlanValidator()
    max_attempts = 5
    last_plan: TaskPlan | None = None

    for attempt in range(1, max_attempts + 1):
        logger.info("Attempt %d to generate TaskPlan", attempt)

        result = agent.run_sync(
            """
## Role:
You are an assistant Game Master and Game Source Material Writer for the Pathfinder Remastered Role Playing Game.  You are familiar with the world of Golarion.
## Intent:
Generate source material for the Sandpoint Hinterlands
## Context:
The players are currently playing a conversion of the old Rise of the Runelords Adventure Path, converted to Pathfinder Remastered.
They are playing in the Sandpoint Hinterlands.
Locations of the sandpoint hinterlands:

* The Pyre
* The Three Cormorants
* The Old Light
* Tickwood
* Shank's Wood

## Control:

Must produce 1 document for each location.  documents must be markdown syntax.

## Output:
Each location must have the following sections in the document:

* Location Overview. Purpose: provide a quick, evocative summary
* Geography & Environment. Purpose: Ground the location int he world's physical reality
* Notable Features. Purpose: Identify key areas a party may explore.

Each location's document should be a "chapter" long, written at the quality and detail for commercial sale.

Each location's document must be formatted with markdown saved in a file for each location.

    """
        )

        usage = result.usage()
        logger.info(calculate_cost(usage))
        logger.info("Took %s %s", usage.requests, p.plural("try", usage.requests))

        plan: TaskPlan = result.output
        last_plan = plan

        # Validate the generated TaskPlan before using it
        if validator.validate(plan):
            logger.info("Successfully generated a valid TaskPlan on attempt %d", attempt)
            break
        else:
            logger.warning(
                "Generated TaskPlan failed validation on attempt %d; retrying if attempts remain",
                attempt,
            )
    else:
        # If we exit the loop without breaking, all attempts failed
        logger.error(
            "Failed to generate a valid TaskPlan after %d attempts", max_attempts
        )
        raise RuntimeError(
            f"Generated TaskPlan is invalid after {max_attempts} attempts according to TaskPlanValidator"
        )

    # At this point, `plan` is guaranteed to be the last valid TaskPlan
    plan = last_plan
    if plan is None:
        raise RuntimeError("No TaskPlan was generated")

    logger.info("Objective: %s", plan.objective)
    logger.info("")
    logger.info("Requires %s %s", len(plan.tasks), p.plural("task", len(plan.tasks)))
    for task in plan.tasks:
        logger.info("Task: %s", task.id)
        logger.info("Prompt:")
        logger.info("```")
        logger.info("%s", task.prompt)
        logger.info("```")
        for dep in task.dependsOn:
            logger.info("Depends on: %s", dep.taskId)
            for i in dep.inputs:
                logger.info("Input (%s): %s", i.type, i.description)
        for output in task.outputs:
            logger.info("Output (%s): %s", output.type, output.description)
        logger.info("")

    # Execute the validated TaskPlan using TaskPlanExecutor and DelegateRunner
    logger.info("Executing TaskPlan with %s", TaskPlanExecutor.__name__)
    delegate_runner = DelegateRunner()
    executor = TaskPlanExecutor(plan, delegate_runner)
    executor.execute()

    # Log a brief summary of execution results
    logger.info(
        "Execution complete. Collected results for %s %s.",
        len(executor.results),
        p.plural("task", len(executor.results)),
    )
    for task_id, result in executor.results.items():
        logger.info("Result for task '%s':", task_id)
        logger.info("  Output types: %s", result.output_types)
        logger.info("  Outputs: %s", result.outputs)


if __name__ == "__main__":
    main()
