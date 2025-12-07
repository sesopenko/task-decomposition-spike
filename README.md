# Task Decomposition Spike

This repository is a small prototype that explores how to represent a **graph of tasks** that are executed by AI agents, where:

- Each **task** is completed by an LLM-backed agent.
- The **outputs** of one task become the **inputs** of other tasks.
- Tasks are connected via explicit **dependencies**, forming a task graph.
- The goal is **demonstration and experimentation**, not production readiness.

## Core Idea

The objective of this project is to:

> Produce a graph of tasks which are completed by agents, feeding the outputs of agents into the inputs of other agents.

Concretely:

- You define a high-level **objective**.
- An LLM-based **planner agent** decomposes that objective into a structured **`TaskPlan`**.
- The `TaskPlan` is a graph of **`Task`** objects, each with:
  - A unique `id`
  - A `prompt` for the agent to run
  - A list of `dependsOn` relationships
  - Typed `inputs` and `outputs`
- The intent is that another layer (not yet implemented here) could:
  - Topologically sort the tasks
  - Execute them in dependency order
  - Pass outputs from upstream tasks into the inputs of downstream tasks.

This repository focuses on **defining the schema** and **using an LLM to generate a plan** that fits that schema.

## Technologies Used

- **Python 3.12+**
- **[pydantic-ai](https://github.com/pydantic/pydantic-ai)** for:
  - Defining structured outputs (`TaskPlan`, `Task`, etc.)
  - Running an LLM agent with a typed output model
- **OpenAI-compatible model** (configured as `gpt-5.1` in `main.py`)
- A simple **cost calculator** (see `cost_calculator.py` in your repo) to estimate LLM usage cost.

## Licensed MIT

This software is licensed under the MIT license, in [LICENSE.txt](LICENSE.txt)

## Copyright

This software is copyright (c) Sean Esopenko 2025
