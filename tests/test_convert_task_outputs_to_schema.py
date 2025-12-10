import json

import pytest

from task_decomposition.models import Task, Output


class TestOutputsToSchema:
    def test_outputs_to_schema_empty_outputs(self):
        """
        When a Task has no outputs, OutputsToSchema should return a JSON schema
        for an object with no properties and no required fields.
        """
        task = Task(
            id="no_outputs",
            prompt="Role: X\nIntent: Y\nContext: Z\nConstraints: C\nOutput: O\n",
            outputs=[],
        )

        schema_str = task.OutputsToSchema()
        schema = json.loads(schema_str)

        assert schema["type"] == "object"
        assert schema["properties"] == {}
        # required may be omitted or empty; normalize to list
        required = schema.get("required", [])
        assert required == []
        # additionalProperties is often false for closed schemas, but we don't
        # assert it here to avoid over-constraining the implementation.

    def test_outputs_to_schema_single_output(self):
        """
        A single Output should become a single required property in the schema,
        with key 'item_0'.
        """
        task = Task(
            id="single_output",
            prompt="Role: X\nIntent: Y\nContext: Z\nConstraints: C\nOutput: O\n",
            outputs=[
                Output(
                    description="A concise summary of the article",
                    type="string",
                )
            ],
        )

        schema_str = task.OutputsToSchema()
        schema = json.loads(schema_str)

        assert schema["type"] == "object"
        properties = schema["properties"]
        assert isinstance(properties, dict)
        assert len(properties) == 1

        # Key must be item_0
        assert "item_0" in properties
        prop_schema = properties["item_0"]
        assert prop_schema["type"] == "string"
        assert prop_schema["description"] == "A concise summary of the article"

        required = schema.get("required", [])
        assert isinstance(required, list)
        # The single property should be required and named item_0
        assert required == ["item_0"]

    @pytest.mark.parametrize(
        "output_type, expected_json_type",
        [
            ("string", "string"),
            ("integer", "integer"),
            ("float", "number"),
            ("boolean", "boolean"),
        ],
    )
    def test_outputs_to_schema_type_mapping(self, output_type, expected_json_type):
        """
        Each Output.type should be mapped to the correct JSON Schema type,
        and the key should be item_0 for a single output.
        """
        description = f"field of type {output_type}"
        task = Task(
            id=f"task_{output_type}",
            prompt="Role: X\nIntent: Y\nContext: Z\nConstraints: C\nOutput: O\n",
            outputs=[
                Output(
                    description=description,
                    type=output_type,  # type: ignore[arg-type]
                )
            ],
        )

        schema_str = task.OutputsToSchema()
        schema = json.loads(schema_str)

        assert schema["type"] == "object"
        properties = schema["properties"]
        assert len(properties) == 1
        assert "item_0" in properties

        prop_schema = properties["item_0"]
        assert prop_schema["type"] == expected_json_type
        assert prop_schema["description"] == description

        required = schema.get("required", [])
        assert required == ["item_0"]

    def test_outputs_to_schema_multiple_outputs(self):
        """
        Multiple outputs should become multiple required properties in the schema,
        with keys item_0, item_1, ..., item_n-1 in order.
        """
        outputs = [
            Output(description="Title of the article", type="string"),
            Output(description="Word count of the article", type="integer"),
            Output(description="Average reading time in minutes", type="float"),
            Output(description="Whether the article is technical", type="boolean"),
        ]

        task = Task(
            id="multiple_outputs",
            prompt="Role: X\nIntent: Y\nContext: Z\nConstraints: C\nOutput: O\n",
            outputs=outputs,
        )

        schema_str = task.OutputsToSchema()
        schema = json.loads(schema_str)

        assert schema["type"] == "object"
        properties = schema["properties"]
        assert isinstance(properties, dict)
        assert len(properties) == len(outputs)

        # Keys must be item_0..item_{n-1} in order
        expected_keys = [f"item_{i}" for i in range(len(outputs))]
        assert sorted(properties.keys()) == sorted(expected_keys)

        json_type_map = {
            "string": "string",
            "integer": "integer",
            "float": "number",
            "boolean": "boolean",
        }

        # Check each item_i maps to the corresponding Output by index
        for i, output in enumerate(outputs):
            key = f"item_{i}"
            assert key in properties
            prop_schema = properties[key]
            assert prop_schema["description"] == output.description
            assert prop_schema["type"] == json_type_map[output.type]

        required = schema.get("required", [])
        # All properties should be required and match the item_i keys
        assert sorted(required) == sorted(expected_keys)
