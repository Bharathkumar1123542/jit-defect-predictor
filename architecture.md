# Architecture — Just-in-Time (JIT) Defect Risk Predictor

## 1. System Context

```
                         ┌─────────────────────────────┐
                         │   Target Open-Source Repo    │
                         │   (e.g. apache/commons-lang) │
                         └───────────────┬──────────────┘
                                         │ (offline, one-time / scheduled)
                                         ▼
┌────────────────────────────────────────────────────────────────────┐
│                        OFFLINE PIPELINE (Day 1)                     │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │  1. Miner     │──▶│ 2. SZZ Labeler│──▶│ 3. Trainer   │──▶ model.pkl│
│  │ (PyDriller)   │   │              │   │ (sklearn/XGB)│   metrics.json│
│  └──────────────┘   └──────────────┘   └──────────────┘            │
│         │                    │                   │                  │
│         ▼                    ▼                   ▼                  │
│   commits.parquet     labels.parquet      model_artifacts/          │
└────────────────────────────────────────────────────────────────────┘
                                         │
                                         │ (model artifact committed to repo /
                                         │  published as a release asset)
                                         ▼
┌────────────────────────────────────────────────────────────────────┐
│                       ONLINE COMPONENTS (Day 2)                     │
│                                                                      │
│  ┌────────────────────────────┐      ┌───────────────────────────┐ │
│  │  GitHub Action              │      │  Dashboard (static site)  │ │
│  │  .github/workflows/         │      │  reads metrics.json +     │ │
│  │  risk-score.yml             │      │  eval_report.json         │ │
│  │  → extract_pr_features.py   │      │  → renders precision/     │ │
│  │  → score.py (loads model)   │      │    recall/AUC/confusion   │ │
│  │  → post PR comment via API  │      │    matrix, feature        │ │
│  └────────────────────────────┘      │    importances             │ │
│                                        └───────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

## 2. Component Breakdown

### 2.1 Miner (`pipeline/mine.py`)
- **Input**: local clone of target repo (shallow or full, date-bounded).
- **Tool**: PyDriller `Repository` traversal API.
- **Output**: `data/commits.parquet` — one row per commit with raw + derived features (§3).
- **Responsibility**: walk commit history once, extract raw diff stats and metadata. No labeling logic here — single responsibility.

### 2.2 SZZ Labeler (`pipeline/label_szz.py`)
- **Input**: `data/commits.parquet` + repo clone (needs `git blame` access).
- **Algorithm**: SZZ — for every commit whose message matches a bug-fix pattern (regex over conventional keywords: `fix`, `bug`, `defect`, plus issue-tracker ID patterns like `[A-Z]+-\d+` for Jira-style, `#\d+` for GitHub issues), run `git blame` on the lines changed by the fix to find the commit(s) that last touched those lines — those are the **inducing commits**, labeled `is_bug_inducing = 1`.
- **Refactoring filter**: exclude fix-commits where diff is >80% whitespace/comment-only changes (reduces SZZ's known false-positive rate on cosmetic fixes).
- **Output**: `data/commits_labeled.parquet` — adds `is_bug_inducing` (0/1) column and `fix_commit_hash` (provenance/audit trail).

### 2.3 Trainer (`pipeline/train.py`)
- **Input**: `data/commits_labeled.parquet`.
- **Split strategy**: **chronological** — sort by commit date, first 80% = train, last 20% = test. This mirrors real deployment (predicting future commits from past ones) and avoids the temporal leakage that a random split would cause.
- **Models**: Logistic Regression (baseline, `class_weight="balanced"`) and XGBoost (`scale_pos_weight` tuned to class ratio) — both trained, best-AUC-on-test selected and persisted. Logistic regression is retained even if XGBoost wins, because its coefficients double as a human-readable "why is this risky" explanation for the PR comment.
- **Output**: `model_artifacts/model.pkl`, `model_artifacts/scaler.pkl` (if LR), `model_artifacts/metrics.json`, `model_artifacts/feature_importance.json`.

### 2.4 GitHub Action (`.github/workflows/risk-score.yml` + `action/extract_pr_features.py` + `action/score.py`)
- **Trigger**: `pull_request` events `[opened, synchronize]`.
- **Step 1 — Feature extraction**: runs PyDriller (or direct `git diff`/`git log` calls) against **only the PR's diff and the PR author's commit history up to that point** — not a full repo re-mine. This keeps the Action fast (<60s).
- **Step 2 — Scoring**: loads `model.pkl` (checked into repo or pulled from latest GitHub Release asset), computes feature vector, outputs a probability.
- **Step 3 — Comment**: uses `actions/github-script` or a Python `requests` call to the GitHub REST API (`POST /repos/{owner}/{repo}/issues/{pr_number}/comments`) to post/update a single sticky comment (identified by an HTML marker so re-runs edit, not duplicate).
- **Permissions**: workflow requests `pull-requests: write` only; uses the ambient `GITHUB_TOKEN` — no PAT/secret management needed.

### 2.5 Dashboard (`dashboard/`)
- **Form**: static single-page HTML/JS (Chart.js or vanilla SVG) reading `model_artifacts/metrics.json` and `model_artifacts/eval_report.json` — no backend server required for v1. Can be served via GitHub Pages from the same repo.
- **Content**: AUC, precision, recall, F1, confusion matrix, class balance, feature importance bar chart, "last trained on" timestamp + commit range.
- **Rationale for static-over-server**: v1 has no need for live queries — metrics are produced once per training run (batch), so a generated JSON consumed by a static page is the simplest architecture that satisfies G5 without adding a hosting/ops dependency for a 2-day build.

## 3. Feature Schema (JIT metrics — the model's input contract)

All features are computed **as of the commit being scored** — i.e., only information available at commit time, never future information (avoids "look-ahead" leakage).

| Feature | Type | Description |
|---|---|---|
| `la` | int | Lines added |
| `ld` | int | Lines deleted |
| `lt` | int | Lines in files touched, before the change (total file size context) |
| `nf` | int | Number of files touched |
| `ns` | int | Number of subsystems (top-level directories) touched |
| `nd` | int | Number of directories touched |
| `entropy` | float | Distribution of changes across files (Shannon entropy — concentrated vs. scattered edits) |
| `ndev` | int | Number of distinct developers who previously touched the touched files |
| `age` | float | Average time (days) since touched files were last changed |
| `nuc` | int | Number of unique prior changes to the touched files |
| `exp` | float | Author's overall commit experience (total prior commits) |
| `rexp` | float | Author's recent experience (prior commits in last N months, weighted) |
| `sexp` | float | Author's experience in the specific subsystem touched |
| `fix_ratio_author` | float | Author's historical bug-fix-commit ratio (# fix commits / # total commits, prior to this commit) |
| `hour_of_day` | int (0–23) | Commit timestamp hour (local/author timezone if available, else UTC) |
| `is_weekend` | bool | Commit made on Sat/Sun |
| `msg_entropy` | float | Shannon entropy of the commit message text (low entropy / very short messages correlate with rushed commits in JIT-SDP literature) |
| `msg_length` | int | Character length of commit message |

Target: `is_bug_inducing` (0/1), produced by the SZZ Labeler.

> These are the standard Kamei et al. JIT feature set (`la, ld, lt, nf, ns, nd, entropy, ndev, age, nuc, exp, rexp, sexp`) plus two additions specific to this project's scope (`fix_ratio_author`, time-of-day, message entropy) called out explicitly in the requirements.

## 4. Data Flow & Storage

- **Intermediate artifacts**: Parquet (columnar, typed, small) over CSV — avoids dtype-inference bugs on re-load and is git-diffable in size only, not content (artifacts are gitignored except final model + metrics JSON).
- **Model artifact**: committed to the repo under `model_artifacts/` (small — logistic regression or shallow XGBoost model, <1MB) so the Action can load it without a network call or external model registry. This is a deliberate simplicity choice for a 2-day project scope; a production system would use a model registry (MLflow, S3) instead — noted as a scaling consideration in §6.
- **No database**: all state is file-based (Parquet + JSON + pickled model). No JIT need for a DB at this scale (a few thousand commits, one repo, batch retraining).

## 5. Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Mining | PyDriller | Purpose-built Git-mining API; handles diff parsing, blame, and traversal without hand-rolled `git log` parsing |
| Labeling | Custom SZZ implementation (PyDriller `git blame` primitives) | Full control over the refactoring filter; avoids opaque black-box SZZ libraries for a project whose credibility rests on transparent methodology |
| ML | scikit-learn (LogisticRegression) + XGBoost | Both standard, well-understood, fast to train on thousands of rows; no deep learning needed or appropriate for tabular JIT features |
| Data interchange | Parquet + JSON | Typed, compact, tool-agnostic |
| Action runtime | Python 3.11 on `ubuntu-latest` GitHub-hosted runner | No infra to manage; matches pipeline language |
| PR comment posting | GitHub REST API via `requests` (or `actions/github-script`) | Native `GITHUB_TOKEN` auth, no extra secrets |
| Dashboard | Static HTML + Chart.js, served via GitHub Pages | Zero backend, zero hosting cost, fits 2-day scope |

## 6. Scalability & Production Considerations (explicitly noted, not built)

- **Model registry**: swap committed `model.pkl` for a versioned artifact in MLflow/S3 if multi-repo or frequent-retrain use cases emerge.
- **Feature store**: if extended to multiple repos, the per-author/per-file rolling stats (`fix_ratio_author`, `age`, `ndev`) would need incremental computation rather than full-history recompute per training run.
- **Action performance at scale**: for monorepos with very large diffs, feature extraction should be capped/sampled rather than processing every changed line.
- **Retraining cadence**: v1 is manually triggered; a production version would run on a schedule (e.g. weekly) via a separate workflow, publishing a new model artifact + updated dashboard metrics.

## 7. Security Considerations

- Action uses the default `GITHUB_TOKEN`, scoped minimally (`pull-requests: write`, `contents: read`) — no long-lived PAT.
- No user-submitted code is executed — feature extraction only reads diff metadata and git history, never runs PR code.
- Model artifact is deterministic and versioned (commit hash of training run recorded in `metrics.json`) so any risk score can be traced back to the exact model version and training data window that produced it.
