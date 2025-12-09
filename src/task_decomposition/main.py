import logging

from pydantic_ai import Agent

from task_decomposition.cost_calculator import calculate_cost
from task_decomposition.models import TaskPlan
from task_decomposition.task_plan_builder import DefaultTaskPlanAgentBuilder
from task_decomposition.task_plan_validator import TaskPlanValidator
import inflect

p = inflect.engine()
logger = logging.getLogger(__name__)


def main():
    # Basic logging configuration; adjust as needed by the application
    logging.basicConfig(level=logging.INFO)

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

Each location's document must be formatted with markdown.

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


if __name__ == "__main__":
    main()
