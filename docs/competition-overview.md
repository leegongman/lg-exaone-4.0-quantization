# Competition Overview

## Context

This project summarizes work from an LG Aimers AI Hackathon focused on lightweight optimization of `EXAONE-4.0-1.2B`.

The public documentation is based on internal Notion notes, an exported PDF, selected local workspace inspection, and reviewed public GitHub repositories. Official competition evidence has not yet been attached to this clean repository.

## Phase 2

Phase 2 focused on submitting a vLLM-compatible checkpoint without custom vLLM runtime changes.

Known constraints from internal records:

- The submitted checkpoint needed to run in the organizer's vLLM-based environment.
- The evaluation dataset was private.
- The end-to-end evaluation ran under a fixed time limit.
- Performance was judged using quality and inference-speed-related criteria.

Internal note fields record a Phase 2 rank of `19/628`. This is not treated as a verified public claim until official leaderboard evidence is added.

## Phase 3

Phase 3 allowed a quantized checkpoint and a custom vLLM wheel, making runtime-level optimization possible.

Known constraints from internal records:

- The submission could include a custom vLLM wheel.
- The private evaluation included quality, speed, and model-size-related criteria.
- Internal notes describe the scoring mix as accuracy, speed, and model size.
- The end-to-end evaluation ran under a fixed time limit.

Internal note fields record a Phase 3 rank of `26/27`. This is not treated as a verified public claim until official leaderboard evidence is added.

## Result Classification

This repository uses the following result labels:

| Label | Meaning |
|---|---|
| Competition score | Official leaderboard or organizer-confirmed result |
| Internal record | Notion, PDF, or team-maintained record without attached official proof |
| Local benchmark | Result from local scripts or public benchmark tasks |
| Failed or partial experiment | Attempted method without conclusive competition impact |

## Public Wording Rule

Until official evidence is attached, Phase 2 and Phase 3 rankings should be described only as internal records. Local benchmark numbers should not be described as competition scores.
