from pydantic_ai import Agent

from task_decomposition.cost_calculator import calculate_cost
from task_decomposition.models import TaskPlan


def main():
    agent = Agent(
        model='gpt-5.1',
        output_type=TaskPlan,
        system_prompt=(
            'Role:'
            'You are a Task Decomposition Planner. Your job is to transform any user request—simple or complex—into a clear, complete, and logically ordered set of subtasks.'
            ''
            'Behavior Requirements:'
            '1. Break the user’s request into the smallest meaningful subtasks needed to accomplish the goal.'
            '2. Ensure the subtasks are actionable—each should describe what needs to be done, not how the final answer should look.'
            '3. Order the subtasks logically, showing dependencies when relevant.'
            '4. Identify required inputs or missing information, and include subtasks for gathering them.'
            '5. Do not perform the subtasks yourself. Only generate the plan.'
            '6. Tasks that depend on other tasks must be clearly defined via dependsOn for the given task'
            '7. The input of a task must match the output of a task which it depends on.'
            
            'Primary Output:'
            'A fully decomposed task plan'
            'Never output anything except the task plan'
        ),
    )
    result = agent.run_sync("""
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

Must produce 1 document for each location.

## Output:
Each location must have the following sections in the document:

* Location Overview. Purpose: provide a quick, evocative summary
* Geography & Environment. Purpose: Ground the location int he world's physical reality
* Notable Features. Purpose: Identify key areas a party may explore.

Each location's document must be formatted with markdown.

    """)
    usage = result.usage()
    print(calculate_cost(usage))

    plan: TaskPlan = result.output
    print(f"Objective: {result.output.objective}")
    print()
    for task in plan.tasks:
        print(f"Task: {task.id}")
        print(f"Prompt:")
        print("```")
        print(task.prompt)
        print("```")
        for dep in task.dependsOn:
            print(f"Depends on: {dep}")
        for input in task.inputs:
            print(f"Input ({input.type}): {input.description}")
        for output in task.outputs:
            print(f"Output ({output.type}): {output.description}")
        print()


if __name__ == "__main__":
    main()
