# Leaderboard submissions

Auditable log of every upload to the challenge portal, newest first. Budget: **5 per day,
15 in total** (25–27 Sep 2026, IST). Uploads are only for versions shortlisted by local
F0.5, never for exploring ideas (`.claude/rules/project-rules.md`, section 3).

## Budget

| Day | Date | Used | Left |
|---|---|---|---|
| 1 | 2026-09-25 | 0 | 5 |
| 2 | 2026-09-26 | 0 | 5 |
| 3 | 2026-09-27 | 0 | 5 |

## Before every upload

1. The version has a row in `experiments/experiments.csv` with its local F0.5 on the fixed
   validation split, and its commit has no `-dirty` suffix.
2. Its notebook, `metrics.json` and code are committed. Note the commit hash.
3. `make validate` passes (our checker and the organisers' validator).
4. Add the entry below before uploading, then fill in the public score.
5. Record the score in the registry: `make public V=vNNN SCORE=0.xxxx`.

## Entry template

```markdown
## Submission NN: YYYY-MM-DD HH:MM IST
- Version: vNNN_<slug>
- Commit: <short hash>
- Change vs previous submission: ...
- Model: ...
- Blocking: ...
- Threshold / decision rule: ...
- Local F0.5 (validation fold): 0.xxxx
- Public F0.5: 0.xxxx
- Notes: ...
```

<!-- Add submissions below this line, newest first. -->
