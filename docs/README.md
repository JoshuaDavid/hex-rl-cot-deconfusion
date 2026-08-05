# Field guide to the experiments

This directory tells you, in simple language, what each experimental arm does.
Each page uses short sentences and concrete examples. An AI model (Claude
Fable 5) wrote these pages and did the experiments. Joshua David owns the
repository and gives direction.

Read the pages in this order:

1. [The task and the reward](the-task.md) — what the model must do, and how we
   score each answer. Read this page first. All arms use this task.
2. [Arm A — vanilla RL](arm-a-vanilla.md) — plain GRPO on random positions.
   Status: complete. Result: the model improved, then hit a wall.
3. [Arm B — staged curriculum](arm-b-staged.md) — fixed curriculum stages.
   Status: retired before a full run. Its parts moved into arm C.
4. [Arm C — adaptive curriculum](arm-c-adaptive.md) — a controller changes the
   task mixture during the run. You can add or remove task types without a
   restart. Status: in smoke test.

For the full decision history, read [RESEARCH_LOG.md](../RESEARCH_LOG.md) in
the repository root. That file is append-only. For the research questions
(C1 to C4), read [RESEARCH_AGENDA.md](../RESEARCH_AGENDA.md).
