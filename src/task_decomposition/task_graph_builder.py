from src.task_decomposition.models import TaskPlan



class TaskGraphBuilder:
    _taskPlan: TaskPlan
    def __init__(self, task_plan: TaskPlan):
        self._taskPlan = task_plan

    def get_sorted_id_list(self, id_list: list[str]):
        sorted_id_list: list[str] = []
        raise Exception("Not implemented")
        return sorted_id_list
