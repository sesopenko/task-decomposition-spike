from pydantic_ai import Agent

from task_decomposition.cost_calculator import calculate_cost
from task_decomposition.models import TaskPlan
import inflect

p = inflect.engine()

def main():
    agent = Agent(
        model='gpt-5.1',
        retries=5,
        output_type=TaskPlan,
        system_prompt=(
            "Role:\n"
            "You are a Task Decomposition Planner. You do NOT solve the user's request directly. "
            "Instead, you design a graph of tasks that will be executed by separate delegate agents.\n\n"

            "Architecture & Purpose:\n"
            "- The user provides a high-level objective.\n"
            "- You break this objective into a set of smaller, well-defined tasks.\n"
            "- Each task will be executed by a separate LLM-backed delegate agent.\n"
            "- Delegate agents have limited context: they only see their own task prompt and the "
            "structured inputs passed into context from upstream tasks as structured objects extending BaseModel from the Pedantic AI python library..\n"
            "- Your goal is to design tasks so that these smaller-context agents can collectively "
            "produce a better result than a single monolithic prompt.\n\n"

            "Behavior Requirements:\n"
            "1. Decompose the user's request into the smallest meaningful subtasks needed to "
            "   accomplish the overall objective.\n"
            "2. Each task must be actionable and self-contained: its prompt must clearly state "
            "   what the delegate should do, not the final answer itself.\n"
            "3. Explicitly model dependencies between tasks using dependsOn.\n"
            "4. If a task needs information produced by another task, it MUST depend on that task, "
            "   and its dependency.inputs must correspond to the upstream task's outputs.\n"
            "5. Design outputs to be machine-consumable and typed, using the allowed types "
            "   (string, integer, float, boolean). Keep outputs as small and focused as possible "
            "   while still being sufficient for downstream tasks.\n"
            "6. Do NOT perform the subtasks yourself. Do NOT write the final documents or content. "
            "   Only generate the task plan.\n"
            "7. The input of a task must match the output of a task which it depends on, both in "
            "   type and in semantic meaning.\n\n"

            "Task Prompt Format:\n"
            "Each Task.prompt MUST follow this structure:\n"
            "- Role: who the delegate agent is (e.g., 'You are a setting writer for Golarion...').\n"
            "- Intent: what this specific task must accomplish.\n"
            "- Context: all information the delegate needs, including any inputs from dependencies.\n"
            "- Constraints: style, format, length, rules, or other limitations.\n"
            "- Output: a precise description of what the delegate must return, aligned with the "
            "  Task.outputs definitions.\n\n"

            "Output Requirements:\n"
            "- Your only output is a TaskPlan object.\n"
            "- The TaskPlan must contain:\n"
            "  * objective: a clear restatement of the user's overall goal.\n"
            "  * tasks: a list of Task objects forming a valid dependency graph.\n"
            "- Every dependsOn.taskId must reference an existing task.id.\n"
            "- Every Dependency.inputs must be consistent with the referenced task's outputs.\n"
            "- Never output anything except the TaskPlan.\n"
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

Must produce 1 document for each location.  documents must be markdown syntax.

## Output:
Each location must have the following sections in the document:

* Location Overview. Purpose: provide a quick, evocative summary
* Geography & Environment. Purpose: Ground the location int he world's physical reality
* Notable Features. Purpose: Identify key areas a party may explore.

Each location's document must be formatted with markdown.

    """)
    usage = result.usage()
    print(calculate_cost(usage))
    print(f"Took {usage.requests} {p.plural("try", usage.requests)}")

    plan: TaskPlan = result.output
    print(f"Objective: {result.output.objective}")
    print()
    print(f"Requires {len(plan.tasks)} {p.plural('task', len(plan.tasks))}")
    for task in plan.tasks:
        print(f"Task: {task.id}")
        print(f"Prompt:")
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
