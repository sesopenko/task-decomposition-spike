# Passing Structured Context to Delegate Agents with pydantic-ai

This document explains how to pass **structured context** into a delegate agent using `pydantic-ai`, by providing an instance of a class that extends `pydantic.BaseModel` as an argument to `Agent.run_sync()`.

The goal is to:

- Take the **structured output** of one delegate agent.
- Convert it into a **structured input model** for another delegate agent.
- Pass that model instance into `run_sync()` (`ctx` parameter) so the LLM sees it as structured JSON, not just interpolated text.

This fits naturally into the `TaskPlan` / `TaskPlanExecutor` / `DelegateRunner` architecture in this repository.

---

## Conceptual Overview

In `pydantic-ai`, an `Agent` call looks like this:

