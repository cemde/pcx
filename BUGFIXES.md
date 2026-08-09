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
7. Commit, push, open a PR that says `Closes #NN`.
8. Have a review agent audit it (section 7) before merging.

## 5. Verification

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

| Issue | Branch | Status | PR |
| --- | --- | --- | --- |
| #67 | `fix/67-optim-scale-by` | fixed at `72b6e68`, in review | |
| #68 | `fix/68-mask-negation` | fixed at `3908095`, in review | |
| #69 | `fix/69-vode-prefix-matching` | fixed at `4f1d20d`, in review | |
| #70 | `fix/70-param-protocols` | fixed at `1084fb4`, in review | |

**#67**, one line in `pcx/utils/_optim.py`: `set(g, g * scale_by)` became
`jtu.tree_map(lambda _g: _g * scale_by, g)`, so the scaled value lands in a
fresh `Param` rather than the caller's. Gate 386 passed, catalogue 50 failed.

**#68**, one line in `pcx/utils/_mask.py`: `_M_not.__call__` renamed to
`_M_not.apply`. The negation expression was already correct and `M.__call__` was
already inherited, so the whole defect was the method name it was bound to.
Gate 388 passed, catalogue 48 failed.

**#69**, two lines in `pcx/predictive_coding/_vode.py`: `re.match` became
`re.fullmatch` for the status, and the rule pattern became
`({key}(?::.*)?)$` so the key must be the whole key. Gate 389 passed,
catalogue 47 failed.

**#70**, 62 lines in `pcx/core/_parameter.py`: removed the dead Python 2
`__div__`/`__rdiv__`/`__idiv__`, added ten in-place operators and
`__float__`/`__int__`/`__len__`/`__iter__`, each a single delegation in the
file's existing idiom. Gate 397 passed, catalogue 39 failed. **13 tests, not
12** as an earlier note said.

None of the four are merged or PR'd; all are pending review sign-off.

### Open concerns for the reviewers

- **#69** anchors asymmetrically: `fullmatch` for the status but `re.match`
  plus a `$` in the caller's pattern for the rule, while `Vode.get` builds a
  pattern with no `$`. Also `$` matches before a trailing newline, and
  `(?::.*)?` narrows the accepted grammar, so `z <- u :double` with a space
  would now stop matching if anything uses that spelling.
- **#70** takes `ty` from 237 to **245**, the only breach of the stated bar.
  The claim is that the +8 is the same pre-existing `possibly-unbound-attribute`
  noise the ~60 existing forwards already emit. Must be verified per-diagnostic.
  Separately, adding `__len__` and `__iter__` makes `Param` iterable and sized
  for the first time, so anything duck-typing on those changes behaviour;
  `_make_tuple` in `pcx/functional/_transform.py` is the first place to check.
| #71 | | not started | |
| #73 | | not started | |
| #72 | | not started | |
| #75 | | not started | |
| #76 | | not started | |
| #74 | | not started | |
| #78 | | not started | |
| #79 | | not started | |
| #77 | | not started | |

**Note on the three branches above.** They were created by agents working in git
worktrees under `.claude/worktrees/`. Branch refs are shared with the main
repository, so the branches survive, but if a branch still points at the same
commit as `main` then no work was committed and the issue should be treated as
not started. Check with:

```shell
git worktree list
git log --oneline main..fix/67-optim-scale-by   # empty output means nothing landed
git worktree prune                              # after removing any stale worktree
```

Stale locked worktrees can be removed with
`git worktree remove --force .claude/worktrees/<name>`. Delete an empty branch
with `git branch -D <name>` and start it again.

**Recommended order to resume**, given the sequencing note in section 6: do #67,
#68, #69, #70 in parallel (file-disjoint), then #73, #72, #75, #76 in parallel,
then #71, #74, #78, #77 strictly in sequence because they all edit
`pcx/functional/_transform.py`, and #79 alongside any of them.
