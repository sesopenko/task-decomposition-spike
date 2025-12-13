from typing import Union

from pydantic import BaseModel, Field
from models_schema import Output as TaskOutput

OutputValueType = Union[str, int, float, bool]


class DelegateOutput(BaseModel):
    value: OutputValueType = Field(
        ...,
        description="The concrete value of the delegate output (string, integer, float, or boolean).",
    )

    def ToSchema(self, dependency_output: TaskOutput) -> str:
        """
        Creates JSON Schema for DelegateOutput, useable by agent to understand what's to be output.

        :param dependency_output: the specification of the output
        :return: JSON schema of DelegateOutput
        """

