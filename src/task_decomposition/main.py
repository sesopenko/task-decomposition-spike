from pydantic_ai import Agent

from task_decomposition.cost_calculator import calculate_cost
from task_decomposition.models import TaskPlan
from task_decomposition.task_plan_builder import DefaultTaskPlanAgentBuilder
from task_decomposition.task_plan_validator import TaskPlanValidator
import inflect

p = inflect.engine()


def main():
    # Build the TaskPlan-producing agent via the abstraction
    builder = DefaultTaskPlanAgentBuilder()
    agent: Agent[TaskPlan] = builder.build_agent()

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
    print(calculate_cost(usage))
    print(f"Took {usage.requests} {p.plural('try', usage.requests)}")

    plan: TaskPlan = result.output

    # Validate the generated TaskPlan before using it
    validator = TaskPlanValidator()
    if not validator.validate(plan):
        raise ValueError("Generated TaskPlan is invalid according to TaskPlanValidator")

    print(f"Objective: {plan.objective}")
    print()
    print(f"Requires {len(plan.tasks)} {p.plural('task', len(plan.tasks))}")
    for task in plan.tasks:
        print(f"Task: {task.id}")
        print("Prompt:")
        print("```")
        print(task.prompt)
        print("```")
        for dep in task.dependsOn:
            print(f"Depends on: {dep.taskId}")
            for i in dep.inputs:
                print(f"Input ({i.type}): {i.description}")
        for output in task.outputs:
            print(f"Output ({output.type}): {output.description}")
        print()


if __name__ == "__main__":
    main()
