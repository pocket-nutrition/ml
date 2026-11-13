---
name: Feature request
about: Suggest an idea or improvement for this service
title: "[Feature] "
labels: enhancement
assignees: ''
---

## Is your feature request related to a problem?

A clear and concise description of what the problem is (e.g. "I'm always frustrated
when...").

## Describe the solution you'd like

A clear and concise description of what you want to happen. If it changes the `/predict`
request or response contract (`MealInput`, `PredictionResponse`, `RawProfile`, etc. in
`app/schemas.py`), include an example payload.

## Describe alternatives you've considered

Any alternative solutions or features you've considered.

## Scope check

- [ ] This does not require this service to hold or hardcode nutritional data (see the
      "No hardcoded nutritional data" section in `CLAUDE.md`) — this service is a pure
      cooking-transform function.
- [ ] This does not require this service to query a database — it is intentionally stateless.

## Additional context

Add any other context, references, or screenshots about the feature request here.
