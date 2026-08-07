# Contributing to PCX

## Setup

PCX uses [uv](https://docs.astral.sh/uv/) for packaging and [just](https://github.com/casey/just) as the task runner.

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
git clone https://github.com/liukidar/pcx.git
cd pcx
just install
```

`just install` creates a `.venv` from `uv.lock`, so everyone gets byte-identical dependencies. Run `just` on its own to list every recipe.

## The loop

Work on a branch, and before opening a pull request run:

```shell
just all     # format, auto-fix, check, test
```

That is the same set of gates CI runs, so a green `just all` should mean a green pipeline.

| Gate        | Command           | Tool                                            | Blocking |
| ----------- | ----------------- | ----------------------------------------------- | -------- |
| Format      | `just format`     | [ruff format](https://docs.astral.sh/ruff/formatter/) | yes  |
| Lint        | `just lint`       | [ruff](https://docs.astral.sh/ruff/linter/)     | yes      |
| Type check  | `just typecheck`  | [ty](https://github.com/astral-sh/ty)           | no       |
| Tests       | `just test`       | [pytest](https://docs.pytest.org/)              | yes      |

Ruff's rule selection lives in `pyproject.toml` under `[tool.ruff.lint]`, pinned explicitly so linting does not shift underneath you when ruff releases a new version.

Type checking is **advisory for now**. `pcx` carries a baseline of roughly 115 `ty` diagnostics that come from its dynamic pytree design — modules that rewrite `__dict__` during unflattening, and parameters that forward attribute access to the array they wrap. The CI job reports them but does not fail the build. Please do not add new ones; once the baseline is cleared, the job becomes a hard gate.

## Dependencies

Add packages with `uv`, never by hand-editing the lockfile:

```shell
uv add equinox            # runtime dependency
uv add --group dev pytest # development tooling
uv lock --upgrade         # refresh the lock to the newest allowed versions
```

Commit `pyproject.toml` and `uv.lock` together. A `pyproject.toml` that has drifted from `uv.lock` breaks environment setup for everyone.

## Tests

Tests live in `tests/`, organised in four tiers:

| Directory | What it proves | Oracle |
| --- | --- | --- |
| `tests/core/` | pytree and parameter machinery | structural invariants |
| `tests/functional/` | the jax-wrapper transforms | hand-written raw jax |
| `tests/numerics/` | energies, gradients, optimiser, layers | closed forms, `optax`, bare `equinox` |
| `tests/devices/` | a training step per backend | local only, never CI |

**Derive every expectation from what the code is meant to do, never from what it currently returns.** Write the closed form, or the raw-jax equivalent, or drive `optax` directly, and compare against that. A test written by observing the implementation will faithfully encode its bugs and hand you a green tick over broken behaviour. If a test must pin current behaviour instead of asserting correctness, name it `test_characterises_*` and say so in the docstring.

Tests run against the CPU backend: `tests/conftest.py` pins `JAX_PLATFORMS=cpu` before JAX is imported. An autouse fixture reseeds `pcx.RKG` around every test — it is wall-clock-seeded module-level state and the default argument of every layer and Vode constructor, so without that the suite is order-dependent. Use the `key` and `rkg` fixtures for anything that draws randomness.

### Markers

```shell
just test           # default gate: excludes bug and device
just test-bugs      # the known-defect catalogue — expected to fail
just test-devices   # accelerator smokes; absent backends skip
just test-all       # everything
```

`bug` marks a test that asserts correct behaviour a catalogued defect currently violates. They are deliberately **not** `xfail`-ed: `xfail` makes a real defect invisible in the ordinary green run, which is how these bugs survived in the first place. Each carries a one-line rationale and an entry in [BUGS.md](BUGS.md). When you fix a defect, delete both the marker and the entry — the test then guards the fix.

`device` marks tests needing a real accelerator. GitHub Actions has no GPU, so these never run there.

Add tests for any behaviour you add or change. CI runs the suite on Linux, macOS and Windows across Python 3.11–3.13, plus an advisory job that reports the bug catalogue.

GPU code paths cannot be exercised on GitHub Actions. If your change touches them, run `just test-devices` locally and say so in the pull request.

## Changelog

Every user-visible change gets an entry under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md), in the appropriate `### Added` / `### Changed` / `### Fixed` / `### Removed` subsection. Reference the PR number.

Purely internal changes — a refactor with no behavioural effect, a typo in a comment — do not need one.

## Releasing

1. Move the `[Unreleased]` entries into a new `## [X.Y.Z] - YYYY-MM-DD` section in `CHANGELOG.md`.
2. Bump `version` in `pyproject.toml` to the same `X.Y.Z`.
3. Merge, then tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.

The `Release` workflow refuses to publish unless the tag, the `pyproject.toml` version and a matching changelog section all agree. It then builds with `uv build`, verifies the wheel imports on a clean interpreter, publishes to PyPI through [trusted publishing](https://docs.pypi.org/trusted-publishers/), and opens a GitHub release whose notes are that changelog section.

## Pull requests

Add docstrings and comments to what you write. Request Luca as a reviewer, and once approved use **Squash and Merge** to keep the history tidy.

To skip CI on an intermediate commit, start the commit message with `[skip ci]`.
