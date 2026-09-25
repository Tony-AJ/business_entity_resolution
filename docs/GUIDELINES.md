# Guidelines and key instructions (Amazon ML Challenge 2026)

Condensed from the organisers'
`guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf` (kept outside the repo).
When this summary and the PDF disagree, the PDF wins. Task, data and output rules:
[PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md).

## Timeline

- Challenge window: **25 Sep 2026, 00:00 IST to 27 Sep 2026, 23:59 IST**.
- Problem statement and dataset are available on day 1; teams build and submit until day 3.
- A live leaderboard shows rankings during the challenge; the final leaderboard is
  revealed afterwards.

## Submissions

- **At most 5 submissions per day** for the 3 days (15 in total). The submit button is
  disabled afterwards.
- Two leaderboards, public and private. Evaluation and shortlisting use performance on
  both.
- **Keep the version history of all submissions**: shortlisting is based on the submitted
  solutions, and the final source code may be requested later.

## Artefacts for the best solution

- A **1–2 page document** explaining the ML approach, the models used, the experiments and
  the conclusion.
- **Source code for experiments, training and inference, with proper comments describing
  the functions.**

## Top 100 teams

Once artefacts, leaderboard scores and every member's eligibility are confirmed, the top
100 teams are announced and must then submit:

- methodology used;
- candidate generation / blocking strategy;
- model architecture and feature engineering;
- any other relevant information about the approach.

These match the sections of `Documentation_template.md` in the student resource.

## Conduct and access

- Cheating, plagiarism or unfair practice, including registering under multiple IDs, means
  instant disqualification.
- Desktop or laptop only, and one device per participant. Simultaneous logins can make the
  system terminate the attempt.
- Queries go through the organisers' Google Form. For technical problems, clear the browser
  cache, try another browser, incognito mode or another network; otherwise email
  support@unstop.com with a screenshot and the registered email ID. Support does not make
  decisions for teams.

## How the repo meets these

| Requirement | Where |
|---|---|
| Version history of all submissions | `LEADERBOARD.md`, `experiments/experiments.csv`, one commit per version |
| Experiment, training and inference code with function comments | notebooks in `experiments/vNNN_<slug>/`, library in `src/entity_resolution/` |
| 1–2 page methodology document | the student resource's `Documentation_template.md`, filled in at the end |
| 15 uploads in total | local experiments first; only shortlisted versions go to the leaderboard |
