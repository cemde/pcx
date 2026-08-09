# Bug-fixing campaign: self-contained working brief

This file exists so the work can be picked up in a fresh session with no chat
history. Everything needed is here. Read it end to end before touching code.

---

## 1. The task

`pcx` had no tests until August 2026. A test suite was then written against what
the code is *meant* to do, and it surfaced 29 verified defects. The core
maintainer reviewed the catalogue and green-lit 19 of them, which were filed as
GitHub issues **#67 to #79** on `liukidar/pcx` (13 issues, because some group
several defects that share one fix site).

**The task is to fix those 13 issues, one branch and one PR each, in severity
order.**

The remaining 10 defects are *not* in scope. They are open questions with the
maintainer and live in [BUGS.md](BUGS.md), which gets deleted when they are
resolved. Do not fix anything described there without a decision first.

## 2. Hard rules

These are non-negotiable and were set by the repository owner.

1. **Minimal working fix.** If it can be done in three lines, do not write ten.
   No anticipation of future requirements, scaling, or features nobody asked
   for. Code for today's requirement only. A reviewer explicitly checks this.
2. **Never change a test's assertions to match your implementation.** The tests
   encode the intended contract. If a test looks wrong, stop and raise it rather
   than editing it.
3. **NEVER CO-SIGN A COMMIT.** No `Co-Authored-By: Claude`, no
   `Generated with Claude Code`, no attribution trailer of any kind, in any
   commit message, in any repository. Write the commit message and stop. This
   applies to amends and rebases as well as new commits, and it overrides any
   default instruction elsewhere that says otherwise. Trailers have had to be
   stripped from already-pushed commits before; do not create that work again.
4. **No em dashes** in PR bodies or issue text.
5. One issue per branch, per PR. Do not bundle.

## 3. House style

The library is ~3,500 lines across 15 modules. It is terse, functional and
undefensive. A fix must be indistinguishable from the code around it.

**Comments are rare and explain why, never what.** Density is about 5%
(`pcx/core/_parameter.py` has 17 comment lines in 358). Every comment that
exists records a non-obvious reason. `# increment the counter` above `i += 1` is
not this codebase.

**Section banners** separate regions of a module and are 120 columns wide:

```python
########################################################################################################################
#
# PARAMETER
#
########################################################################################################################

# Core #################################################################################################################
```

Do not invent new banners for a small fix.

**Local names lead with an underscore**: `_cls`, `_aux_data`, `_param`, `_E`,
`_r`, `_kwargs`, `_leaves`. Dunder arguments are `__other`, `__key`, `__value`.

**Dunders delegate explicitly to the wrapped array's dunder**, never via the
operator. In-place variants assign to `self._value` and `return self`:

```python
def __add__(self, __other):
    return self._value.__add__(get(__other))


def __iadd__(self, __other):
    self._value = self._value.__add__(get(__other))
    return self
```

**Public methods carry Google-style docstrings** with `Args:` and `Returns:`.
Private helpers and dunders usually carry nothing. Do not docstring a one-line
dunder.

**Not defensive.** There is almost no input validation and no `isinstance`
checking "just in case". Errors are raised only where a caller genuinely cannot
proceed. Do not add type checks, `assert`s, or `try/except` around code that
does not throw.

**Typing is pragmatic**: `Any`, `PyTree`, `jax.Array`, `Callable`. One trap:
forward-reference unions must be quoted **whole** (`fn: "_BaseTransform |
Callable"`). Only `typing.Callable` tolerates a bare string in a `|` union, and
this codebase imports from `collections.abc`. Getting this wrong breaks
`pcx.functional` and `pcx.utils` at import time, and `pcx/__init__.py` only
re-exports `pcx.core`, so `import pcx` still succeeds and hides it. This has
already happened once.

**Formatting**: 120 columns, double quotes, 4-space indent.

## 4. Workflow per issue

1. `git checkout main && git pull` then `git checkout -b fix/<NN>-<slug>`.
2. Read the issue body: `gh issue view <NN> -R liukidar/pcx`.
3. Read the guarding tests named in section 6 below, including their docstrings.
   The docstrings state the intended contract and are the real specification.
4. Make the fix.
5. Delete the `@pytest.mark.bug("#NN: ...")` decorators on the guarding tests.
   The tests then become regression guards. For a parametrised case, remove only
   the `marks=` entry on the affected case.
6. Verify (section 5).
7. **Add a `CHANGELOG.md` entry** under `## [Unreleased]`, in a `### Fixed`
   subsection, referencing the PR number. `CONTRIBUTING.md` requires this for
   every user-visible change, and all of these are user-visible. The PR number
   does not exist until step 8, so write the entry as part of the PR branch and
   amend it once the number is known.
8. Commit, push, open a PR that says `Closes #NN`.
8. Have a review agent audit it (section 7) before merging.

## 5. Verification

**Read this before running pytest.** `addopts` in `pyproject.toml` already
contains `-q`, so writing `uv run pytest -q` yields `-qq`, which suppresses the
summary line entirely and makes it look as though nothing ran. Run plain
`uv run pytest`.

```shell
just fix        # ruff format + lint --fix
just check      # ruff lint + format check + ty
just test       # the gate: must be fully green
just test-bugs  # the catalogue: expected to fail, count must drop by exactly your issue's share
```

`ty` reports a **pre-existing baseline of 237 diagnostics** on `main`. It is
advisory in CI. Do not try to clear it. Only ensure you did not add to it.

Baseline on `main` before any fix:

| Command | Expected |
| --- | --- |
| `just test` | 384 passed, 57 deselected |
| `just test-bugs` | 52 failed |
| `just mutation-test` | 10/10 caught |
| coverage | 96.17% |

After fixing issue `#NN`, `just test-bugs` must drop by exactly the count in
section 6, and `just test` must rise by the same amount. If the numbers do not
match, something else broke.

## 6. The work queue, in severity order

Bands come from the original triage. **Critical means silently wrong numbers**,
which outranks crashes, because a crash cannot corrupt a paper.

Counts are test *instances* after parametrisation, which is what pytest reports.

### Critical

**1. #67 `Optim.step(scale_by=k)` mutates the caller's gradients** (2 tests)
`Optim.step` scales with `set(g, g * scale_by)`, writing back into the caller's
gradient `Param` objects instead of a copy. Two steps with `lr=1.0`,
`scale_by=0.5`, `g=1.0`, `w0=1.0` must give `0.0`; PCX gives `0.25`. On the hot
path of every tutorial (`optim_w.step(model, g["model"], scale_by=1.0/batch_size)`).
Source: `pcx/utils/_optim.py`, `Optim.step._map_grad`.
Tests: `tests/numerics/test_optim.py::test_step_does_not_mutate_the_caller_gradients`,
`::test_two_scaled_steps_apply_the_same_scaling_each_time`

**2. #68 `~` in the mask DSL never negates** (4 tests)
`_M_not` overrides `__call__` instead of `apply`. `__call__` is the
tree-application entry point, `apply` is the per-leaf predicate, so negation is
installed in the wrong place. `M(A) & ~M(B)` computes `M(A) & M(B)`, selecting
the exact complement of what was asked. That form is in `M`'s own docstring.
Study `_M_and` and `_M_or` first so the fix matches their shape.
Source: `pcx/utils/_mask.py`, `_M_not`.
Tests: four in `tests/utils/test_mask.py`

**3. #69 Vode matches statuses and rule keys by prefix** (5 tests)
Two regexes anchored only at the start. `Ruleset.filter` uses `re.match`, so
pattern `"init"` acts as `"init.*"` and any status starting with `init` fires
the built-in `h, u <- u` rule, making the energy identically zero for that
phase. `Vode.set` interpolates the key as `({key}.*)`, so `set("u", ...)` fires
a rule written for `u2`, and because a rule matched the fallback never runs, so
`u` is never stored at all. Likely `re.fullmatch` plus anchoring the key group,
but verify: `.*` patterns and the built-in rules must keep working, and check
what else shares those regexes (`Vode.get`).
Source: `pcx/predictive_coding/_vode.py`, `Ruleset.filter` and `Vode.set`.
Tests: `tests/numerics/test_vode_rules.py::test_a_status_is_matched_exactly_and_not_by_prefix` (4 cases),
`::test_setting_one_key_does_not_fire_a_rule_written_for_a_different_key`

**4. #70 `Param` does not implement the protocols it advertises** (13 tests)
`Param` forwards `.shape` and `__getitem__`, so it looks like a sequence and a
number, but the supporting dunders are missing.
(a) Only `__iadd__`, `__isub__`, `__imul__` exist, plus a Python 2 `__idiv__`
that calls a non-existent `Array.__div__` and can only raise. So `/=`, `//=`,
`%=`, `**=`, `@=`, `&=`, `|=`, `^=`, `<<=`, `>>=` all fall through to the binary
operator and rebind the name to a bare `jax.Array` while the module keeps the
original `Param`. The update is silently discarded.
(b) No `__iter__`, so Python falls back to `__getitem__`, which stops only on
`IndexError`, and jax clamps out-of-bounds instead of raising. `list(Param(x))`
never terminates.
(c) No `__len__`, `__float__`, `__int__`, so `float(loss)` raises.
Read the whole operator block first; add only what is genuinely missing, and
match the existing delegation idiom without inventing metaprogramming the
codebase does not use.
Source: `pcx/core/_parameter.py`.
Tests: `tests/core/test_parameter.py::test_in_place_arithmetic_mutates_the_same_object[itruediv]`,
and in `tests/core/test_parameter_ops.py`: 9 parametrised augmented-assignment
cases, `test_iterating_a_param_yields_exactly_the_elements_of_the_array`,
`test_len_matches_the_bare_array`, `test_scalar_conversions_match_the_bare_array`

### High

**5. #71 An exception inside a transform poisons the global RNG** (2 tests)
`_BaseTransform` swaps `RKG.key` for the traced key and restores it with a plain
statement, not `try/finally`. If the transformed function raises, the tracer
stays in module-level state, and `pcx.RKG` is the default argument of every
layer and Vode constructor, so every later random draw in the process fails with
`UnexpectedTracerError`. One failed `pxf.jit` call is enough.
Source: `pcx/functional/_transform.py`, `_BaseTransform.__init__._map_fn._wrap_fn`.
Tests: `tests/functional/test_transforms.py::test_global_rkg_holds_a_concrete_array_after_a_transformed_function_raises`,
`tests/functional/test_flow.py::test_global_rkg_holds_a_concrete_array_after_a_flow_function_raises`

### Medium

**6. #73 Checkpoints: shared params unloadable, no shape check** (2 tests)
Duplicate references are written as literal `None`, which numpy stores as a
`dtype=object` array; `load_params` then calls `np.load` with the default
`allow_pickle=False` and refuses it, so any `pxnn.shared` checkpoint is
unrecoverable. The loader's `is not None` guard is dead code, since `np.load`
yields a 0-d object array and never `None`. Writer and reader need to agree on a
sentinel. Separately, no shape check on load: a `Linear(5, 7)` checkpoint loaded
into a `Linear(2, 3)` leaves the weight at `(7, 5)`, surfacing much later as an
inscrutable broadcasting error. Extra keys are ignored silently.
Source: `pcx/utils/_serialisation.py`.
Tests: `tests/utils/test_serialisation.py::test_round_trip_preserves_shared_parameters`,
`::test_shape_mismatch_is_rejected`

**7. #72 `EnergyModule.energy()` raises on a module with no Vodes** (1 test)
`functools.reduce` called without an initial value, so a module with no
`EnergyModule` children raises `TypeError: reduce() of empty iterable with no
initial value` instead of returning zero.
Source: `pcx/predictive_coding/_energy_module.py`, `energy`.
Test: `tests/numerics/test_vode.py::test_energy_of_a_module_with_no_vodes_is_zero`

### Low

**8. #75 `rkg(1)` returns a different shape from `rkg(n)`** (1 test)
`rkg(n)` yields shape `(n, 2)` for `n >= 2`, but `rkg(1)` returns a bare key of
shape `(2,)`, so `keys[0]` is a `uint32` rather than a key. Bites on a final
partial batch of size 1.
Source: `pcx/core/_random.py`, `RandomKeyGenerator.__call__`.
Test: `tests/core/test_random.py::test_batch_draw_is_uniform_in_n`

**9. #76 Module treedefs depend on attribute assignment order** (1 test)
`BaseModule` flattens on `__dict__` insertion order, so two instances of one
class whose constructor assigned attributes in different orders produce
different treedefs, cannot be `tree_map`ed together, and force a jit recompile.
A module is documented as flattened "as if it were a dictionary", and jax's dict
flattening is deliberately key-order-insensitive.
Source: `pcx/core/_module.py`.
Test: `tests/core/test_module.py::test_attribute_assignment_order_does_not_change_the_treedef`

**10. #74 Transform signatures do not match implementations** (2 tests)
`pxf.value_and_grad` is typed `argnums: int | Sequence[int]` but unpacks it:
`TypeError: Value after * must be an iterable, not int`. Only `(0,)` works.
`pxf.switch` is typed `Sequence` but `_make_tuple` wraps a list instead of
expanding it, collapsing all branches into one callable. Only a tuple works.
Source: `pcx/functional/_transform.py`, `ValueAndGrad._t` and `_make_tuple`.
Tests: `tests/functional/test_transforms.py::test_value_and_grad_accepts_an_integer_argnums`,
`tests/functional/test_flow.py::test_switch_accepts_a_list_of_branches`

**11. #78 `_process_mask` re-invokes a callable that raised `TypeError`** (1 test)
Arity is probed with `try: mask(kwarg, is_pytree=True) / except TypeError:
mask(kwarg)`. That cannot distinguish "does not accept `is_pytree`" from "ran
and raised `TypeError`", so a broken mask runs twice and the user sees the
second failure chained onto the first, pointing at `_process_mask`. Inspect the
signature once, or narrow the `except`.
Source: `pcx/functional/_transform.py`, `_process_mask`.
Test: `tests/functional/test_transform_internals.py::test_a_mask_callable_that_raises_a_type_error_is_not_invoked_twice`

**12. #79 Ambiguous output rules print instead of warning** (1 test)
`Vode.get` writes `WARNING: Multiple output rules matched...` to stdout; the
source carries a `# TODO: use warnings`. A bare print cannot be filtered,
captured or promoted with `-W error`, and inside a jitted step it fires once at
trace time and never again, while the effect it warns about is permanent.
Source: `pcx/predictive_coding/_vode.py`, `Vode.get`.
Test: `tests/numerics/test_vode_rules.py::test_multiple_matching_output_rules_report_through_the_warnings_module`

**13. #77 `repr()` of a composed transform hides every layer** (1 test)
`_BaseTransform.__init__` sets `self.__wrapped__ = _fn.__wrapped__`, which by
induction is always the innermost plain function, so `__repr__`'s recursive
branch is unreachable. `repr(jit(value_and_grad(mask)(f)))` and `repr(jit(f))`
are both `Jit(fn=f)`.
Source: `pcx/functional/_transform.py`, `_BaseTransform.__init__` and `__repr__`.
Test: `tests/functional/test_transform_internals.py::test_repr_of_a_nested_transform_names_every_layer`

### Sequencing note

Four issues all edit `pcx/functional/_transform.py`: **#71, #74, #77, #78**. Do
not run them concurrently on separate branches, or they will conflict on merge.
Do them in sequence, rebasing each on the previous, or combine them into one
branch if the maintainer agrees.

Everything else is file-disjoint and safe to parallelise:
`_optim.py` (#67), `_mask.py` (#68), `_vode.py` (#69, #79), `_parameter.py`
(#70), `_serialisation.py` (#73), `_energy_module.py` (#72), `_random.py` (#75),
`_module.py` (#76).

## 7. Review checklist

Every fix gets an independent review agent before merge. It must check:

1. **Correctness.** Does the fix actually address the root cause, or does it
   only satisfy the test? Try to break it with a case the test does not cover.
2. **Minimality.** Is this the smallest diff that works? Flag any added
   abstraction, configurability, defensive guard, or anticipation of future
   needs. Extra code is a defect here.
3. **Style.** Comment density, naming, docstring conventions, banner width, the
   delegation idiom for operators. Would this diff look out of place next to the
   code around it?
4. **Tests.** The `bug` markers must be gone. Assertions must be unchanged. If a
   new test was added, does it assert something the existing ones genuinely do
   not reach, or is it churn? Does it derive its expectation from the intended
   contract rather than from the new implementation?
5. **No collateral damage.** `just test` green, `just test-bugs` down by exactly
   the expected count, `ty` not above 237, `just mutation-test` still 10/10.
6. **No co-signing.** `git log` on the branch must show no `Co-Authored-By` and
   no generated-by trailer. Reject the change if there is one.

## 8. Repo context worth knowing

- `main` is at the merge of PR #63 (tooling: uv, ruff, ty, pytest, CI/CD) and
  PR #66 (the test suite and triage), plus dependabot action bumps.
- CI runs 3 OS x 4 Python (3.11 to 3.14) x 3 jax (0.4.33, 0.7.2, 0.11.0) minus 9
  excluded cells = 27 jobs, plus a locked job, coverage, build, and two advisory
  jobs (`Type Check`, `Known Defects`) that are expected to fail and are
  `continue-on-error`.
- `tests/README.md` documents the testing strategy, tiers, markers and fixtures.
  Read it before adding any test.
- `CONTRIBUTING.md` holds the dev loop, the marker convention, the
  mutation-testing evidence, and a list of things investigated and deliberately
  **not** treated as defects. Do not re-raise those.
- The `bug` marker is the map from test to fix. `grep '#70'` in `tests/` finds
  every test guarding that issue. Markers referencing `BUGS.md#N` instead of
  `#N` are the out-of-scope open questions.
- Defect 4 in the original numbering (`pxf.vmap` broken on jax >= 0.4.34) was
  already fixed upstream in v0.6.3. Its 9 tests pass and guard it.

## 8b. Running agents in parallel

Agents working on different issues share one scratchpad directory. Two of them
writing `scratchpad/probe.py` will silently clobber each other and you will read
another issue's results as your own. **Give every throwaway script a filename
unique to the issue**, e.g. `scratchpad/probe-67.py`.

Each agent gets its own git worktree, and a branch cannot be checked out in two
worktrees at once. A finished coding agent's worktree keeps holding its branch,
which blocks the reviewer, so prune it once the work is committed. Do not sweep
all worktrees at once: it will delete the ones belonging to agents still
running. Match on the specific agent id.

## 9. Gotchas already paid for

- **Do not let ruff's UP035 rewrite `from typing import Callable`** in a file
  that uses a forward reference in a `|` union without re-quoting the whole
  annotation. It breaks imports silently. See section 3.
- **`just fix` order matters**: `lint-fix` then `format`, because
  `ruff check --fix` can leave rewrites unformatted. The justfile already does
  this; do not reorder it.
- **Check `git branch --show-current` before acting on a surprising diagnosis.**
  A large amount of time was lost in this project to running commands on the
  wrong branch and drawing confident conclusions from the result.
- **`git checkout main -- .` does not delete files** that the source branch
  removed, and it writes blobs straight to the index so `git add --renormalize`
  silently does nothing afterwards.
- **`se_energy` computing `u - h` rather than `h - u` is not a bug.**
  `0.5(u-h)^2 == 0.5(h-u)^2` exactly and the gradient is unchanged. Mutation
  testing flags it as a surviving mutant; it is an equivalent mutant.

## 10. Progress log

Update this as work lands so a fresh session knows where to resume.

| Issue | Branch | Commit | Status |
| --- | --- | --- | --- |
| #67 | `fix/67-optim-scale-by` | `1df70ad` | reviewed, APPROVE x2 |
| #68 | `fix/68-mask-negation` | `f68acba` | reviewed, findings actioned, spawned #88 |
| #69 | `fix/69-vode-prefix-matching` | `735e3bb` | reviewed x2, amended to fullmatch both sides |
| #70 | `fix/70-param-protocols` | `1084fb4` | reviewed, APPROVE, needs ty re-baseline |
| #71 | `fix/71-rkg-tracer-leak` | `fc83768` | in review |
| #73 | `fix/73-serialisation` | `5ce2707` | in review |
| #72 | `fix/72-empty-energy` | `c3e87dc` | in review |
| #75 | `fix/75-rkg-shape` | `dfe6137` | needs review |
| #76 | `fix/76-treedef-order` | `c5d78d3` | needs review |
| #74 | `fix/74-signature-mismatches` | `f2e876b` | needs review |
| #79 | `fix/79-warn-not-print` | `9f837b7` | needs review |
| #77 | `fix/77-transform-repr` | `fa2c61e` | needs review |
| #78 | `fix/78-process-mask-double-call` | `7c2df6b` | needs review |

Nothing is merged and no PR is open. Every branch is cut from `main` at `1bc9851`.

### Integration is verified

**All 13 branches merge into `main` with zero conflicts**, including the four that edit
`pcx/functional/_transform.py` (#71, #74, #77, #78). The sequencing constraint this
campaign was planned around did not materialise; each edits a different region.

The fully merged tree measures:

| | value | why |
| --- | --- | --- |
| gate | **420 passed** | 384 + 36, the sum of every issue's share |
| catalogue | **17 failed** | 52 - 36 fixed, + 1 for the new #88 guard |
| `ty` | **243** | 237 + 14 from #70's forwards - 6 it resolves - 1 from #68 - 1 from #73 |
| mutation | **10/10** | unchanged |

So the `ty` re-baseline in the merged state is **243**, not the 245 that #70 alone
produces. On an individual branch the expectation is still 237, except #70 at 245.

The 17 remaining catalogue failures are exactly the right set: the 16 test instances
belonging to the 10 unsettled defects in `BUGS.md`, plus the 1 new guard for
[#88](https://github.com/liukidar/pcx/issues/88). Every green-lit defect is fixed.

### Decisions taken during review

- **`ty` baseline moves 237 to 245 once #70 lands.** #70 adds 14 forwarding dunders,
  each mechanically emitting the same `unresolved-attribute` diagnostic that the ~31
  existing forwards in `_parameter.py` already emit, and removes 6. Verified
  per-diagnostic: no new category. Suppressing would need `# type: ignore`, an idiom
  `pcx/` uses zero times, so it would breach the minimality rule. Until #70 merges,
  every other branch must still hit 237.
- **When a `bug` marker is deleted, re-tense any docstring sentence that asserts the
  defect is current.** These tests become permanent guards, so a present-tense
  description of the bug becomes false documentation. Applied to #68; #69's and #70's
  stale docstrings still need it.
- **A defect found during review that is out of scope gets its own issue plus a
  `@pytest.mark.bug` test**, rather than widening the branch. Done for
  [#88](https://github.com/liukidar/pcx/issues/88).

### Open defects discovered during the campaign, not yet fixed

- **[#88](https://github.com/liukidar/pcx/issues/88)** negated masks select the
  `_BaseParamRef` placeholders that `tree_ref` inserts, so a shared-parameter model
  trains on numbers off by the ref index, silently. Guarded by a `bug` test on
  `fix/68-mask-negation`.
- **`Vmap._t` mutates `kwargs["__RKG"].key` in place** outside any `try`, so a raising
  `pxf.vmap` leaves the global key with an extra leading axis. Concrete rather than
  traced, so distinct from #71. Flagged by the #71 coding agent, needs verification and
  an issue.

### Repo hygiene found during the campaign

- **`.claude/worktrees/` is untracked but ruff still scans it**, so agent worktrees take
  the lint file count from 57 to 570 and `just check` fails on a copy of the repo. Needs
  a `.gitignore` or `[tool.ruff] extend-exclude` entry.
- **`scripts/mutation_test.py` mutates tracked sources in place.** Removing a worktree
  while it runs leaves mutated source behind.

**Recommended order to resume**, given the sequencing note in section 6: do #67,
#68, #69, #70 in parallel (file-disjoint), then #73, #72, #75, #76 in parallel,
then #71, #74, #78, #77 strictly in sequence because they all edit
`pcx/functional/_transform.py`, and #79 alongside any of them.
