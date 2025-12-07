from graphlib import TopologicalSorter
from task_decomposition.models import TaskPlan


class TaskGraphBuilder:
    _taskPlan: TaskPlan
    def __init__(self, task_plan: TaskPlan):
        self._taskPlan = task_plan

    def get_sorted_id_list(self):
        sorted_id_list: list[str] = []
        topology = {}
        for task in self._taskPlan.tasks:
            deps: list[str] = []
            for dep in task.dependsOn:
                deps.append(dep.taskId)
            topology[task.id] = deps
        ts: TopologicalSorter = TopologicalSorter(topology)
        sorted_id_list = list(ts.static_order())
        return sorted_id_list
