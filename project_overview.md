# Project Overview — Just-in-Time (JIT) Defect Risk Predictor

## 1. Problem Statement

Code review capacity is finite and does not scale with commit volume. Reviewers currently allocate attention uniformly (or by gut feel) across incoming pull requests, with no data-driven signal for *which specific commits* are statistically likely to introduce a defect. This causes two failure modes:

- **Under-scrutiny of risky commits** — large, late-night, single-author changes to historically bug-prone files get the same review depth as a one-line typo fix.
- **Over-scrutiny of safe commits** — reviewer time is spent re-reading low-risk, mechanical changes.

**Just-In-Time (JIT) Software Defect Prediction** is an established research area (Kamei et al., 2013, "A Large-Scale Empirical Study of Just-in-Time Quality Assurance"; the SZZ algorithm, Śliwerski/Zimmermann/Zeller 2005) that predicts defect-proneness **at the commit level, at commit time**, using change metadata rather than static code metrics. This project implements a working, empirically-validated instance of that research: a classifier trained on real mined commit history, wrapped in a GitHub Action that posts a risk score on every new PR, backed by a dashboard reporting the model's actual precision/recall on a known open-source repo.

## 2. Goals

| # | Goal | Success Criterion |
|---|------|--------------------|
| G1 | Mine commit-level JIT features from a real open-source repo | ≥2,000 labeled commits extracted via PyDriller |
| G2 | Label commits as bug-inducing using SZZ | SZZ-linked bug-fix → inducing-commit mapping produced and stored |
| G3 | Train a defect-risk classifier | Model achieves AUC ≥ 0.65 on held-out test set (JIT literature baseline range is 0.65–0.75 AUC) |
| G4 | Automate risk scoring on new PRs | GitHub Action comments a calibrated risk score + top contributing factors within 60s of PR open/sync |
| G5 | Provide model transparency | Dashboard shows precision/recall/AUC/confusion matrix on the evaluation repo, refreshed per training run |
| G6 | Ship in 2 days | Day 1: mining/labeling/training complete and reproducible. Day 2: Action + dashboard + README live |

## 3. Non-Goals (explicitly out of scope)

- Static code analysis (AST-level, complexity metrics like McCabe/Halstead) — this is a **change-metadata** model, not a code-quality-metrics model. (Explicitly separable future extension — see §7.)
- Multi-repo transfer learning / cross-project prediction.
- Real-time model retraining on every PR (retraining is a manual/scheduled batch job).
- Blocking merges automatically — the Action **comments**, it does not gate CI or set required-status-check failures in v1.
- Support for non-Git VCS.
- Multi-language AST parsing — features are language-agnostic (line/file/author/time/message stats), so the same pipeline works on the JS or Java target repo without per-language tooling.

## 4. Target Users

- **Primary**: engineering teams doing PR-based code review who want a triage signal, not a gate.
- **Secondary**: reviewers of *this* project (open-source contributors, hackathon judges, hiring reviewers) evaluating it as a demonstration of applied SE-research competence.

## 5. Evaluation Repository Selection

One mid-size, actively-maintained repo with a clean linked-issue/bug-fix commit convention, chosen at kickoff and fixed for the life of the project (do not change repos mid-build — SZZ linkage quality depends on consistent commit message conventions). Two acceptable candidates, evaluated in this order:

1. **`apache/commons-lang`** (Java) — long history, conventional `git log` bug-fix references (`LANG-####`), stable, well-scoped subset available via shallow-clone + date range.
2. **A mid-size JS repo with a strict Conventional Commits history** (e.g. `expressjs/express` or `axios/axios`) — fallback if commons-lang's issue-linkage density is too low in the sampled window.

Selection is finalized in Phase 0 of `implementation.md` based on a 10-minute linkage-density probe (see Day 1, Step 1).

## 6. Success Metrics (project-level, reported in README)

- Reported **AUC, Precision, Recall, F1** on a **time-based** train/test split (not random split — random splitting on commit history leaks future information into training, which is a known JIT-SDP methodology error to avoid).
- Class balance reported (bug-inducing commits are typically 10–20% of history — this is an imbalanced classification problem and must be stated as such).
- Action median comment latency.
- Dashboard load time and data freshness timestamp.

## 7. Future Extensions (out of scope for v1, noted for credibility)

- Combine JIT (change) features with static code metrics for a hybrid model.
- Confidence-based CI gating (block only high-confidence, high-impact risk commits).
- Per-file risk heatmap instead of per-commit only.
- Online learning: incrementally update the model as new bug-fix links appear.

## 8. Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| SZZ labeling noise (SZZ has known false-positive issues with cosmetic/refactor changes) | Use "SZZ with refactoring filter" — exclude fix-commits whose diff is >80% whitespace/formatting; document limitation in README rather than over-engineer a fix |
| Small/imbalanced dataset → unstable AUC | Use stratified time-based CV, report confidence via repeated evaluation, prefer XGBoost with `scale_pos_weight` or logistic regression with `class_weight="balanced"` |
| GitHub Action rate limits / auth | Use `GITHUB_TOKEN` built into Actions context; cache dependencies; keep the Action's own compute (feature extraction for a single PR diff) lightweight — no full repo re-mine per PR |
| Reviewers dismissing this as "just a script" | README leads with methodology (SZZ, JIT-SDP citation), reported AUC on a named public repo, and a precision/recall dashboard — signals rigor, not a toy |
