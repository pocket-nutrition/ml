---
name: Bug report
about: Report incorrect predictions or unexpected service behavior
title: "[Bug] "
labels: bug
assignees: ''
---

## Describe the bug

A clear and concise description of what is wrong.

## Request payload

The exact JSON body you sent to `/predict` (or `/health`):

```json
{
  "cooking_type": "",
  "ingredients": []
}
```

## Actual response

The exact JSON response you received.

```json
```

## Expected response

What you expected instead, and why (e.g. a reference value from Ciqual / Open Food Facts, or
expected structural shape).

## Environment

- Service version / commit: `git rev-parse HEAD`
- Python version: `python --version`
- How you ran it: `uvicorn` locally / Docker image / other
- `MODEL_PATH` set and whether a model artifact was present at that path

## Additional context

Add any other context about the problem here (logs, stack traces, etc).
