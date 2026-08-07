# Known defects

Every entry below was reproduced directly and has a failing test that asserts the
correct behaviour. Run them with:

```shell
just test-bugs          # the whole catalogue
uv run pytest -m bug -q # same thing
```

The default test run excludes these (`-m "not (device or bug)"`) so the suite
stays a usable gate. When a defect is fixed, delete its `@pytest.mark.bug` and
the entry here — the test then guards the fix.

Severity is about consequences for a user's results, not about how hard the fix
looks. **Silent** means wrong numbers with no error; those rank above crashes,
because a crash cannot corrupt a paper.

---

## Critical — silently wrong numbers

### 1. `Optim.step(..., scale_by=k)` mutates the caller's gradients

`Optim.step` scales gradients with `set(g, g * scale_by)`, which writes back into
the gradient `Param` objects rather than working on a copy. Step a gradient tree
twice and the scaling compounds.

With `lr=1.0`, `scale_by=0.5`, `g=1.0`, `w₀=1.0`, two steps must give
`1 - 2·(1.0·0.5·1.0) = 0.0`. PCX gives **0.25** — the gradient is 0.5 on the
second step and 0.25 on the third.

This is on the hot path of every tutorial, which all call
`optim_w.step(model, g["model"], scale_by=1.0/batch_size)`. Any loop that feeds
one gradient tree to two optimisers — the standard weights-and-states split in
predictive coding — shrinks its effective learning rate geometrically with no
symptom other than slow convergence.

*Tests:* `tests/numerics/test_optim.py::test_two_scaled_steps_apply_the_same_scaling_each_time`,
`::test_step_does_not_mutate_the_caller_gradients`
*Source:* `pcx/utils/_optim.py`, `_map_grad`

### 2. Shared submodules are double-counted in the energy

`BaseModule.submodules` yields one module once per reference rather than once per
unique object, and `EnergyModule.energy` reduces over that. A module reachable
through two attributes contributes its energy twice: a child of energy 1.0 makes
its parent report **2.0**.

A tied Vode is therefore over-weighted in the objective and in its gradient. The
numbers stay finite and plausible, which is exactly what makes this dangerous.
`tree_ref` deduplicates on `id()` for this reason, so the two halves of the
library disagree about what sharing means.

*Test:* `tests/core/test_module.py::test_submodules_yields_each_module_once_even_when_referenced_twice`
*Source:* `pcx/core/_module.py`, `submodules`; `pcx/predictive_coding/_energy_module.py`, `energy`

### 3. `p /= x` silently replaces the `Param` and loses the update

`Param` defines the Python-2 `__idiv__` (which calls a non-existent
`Array.__div__`) but no `__itruediv__`. So `p /= x` falls through to
`__truediv__` and **rebinds the name to a bare `jax.Array`**. The module still
holds the original `Param`, so the division is silently discarded.

`+=`, `-=` and `*=` all mutate in place correctly, so nothing signals that `/=`
is different. In-place mutation preserving object identity is the invariant the
library's whole update-tracking rests on.

*Test:* `tests/core/test_parameter.py::test_in_place_arithmetic_mutates_the_same_object[itruediv]`
*Source:* `pcx/core/_parameter.py`

---

## High — corrupts process or transform state

### 4. `pxf.vmap` is broken on jax >= 0.4.34

`Vmap._t` calls `_process_mask` without an `rkg_mask`, leaving
`mask["__RKG"]` as `None`. jax treats `None` as an empty pytree node and, since
0.4.34, rejects it as a prefix against the non-`None` `RandomKeyGenerator`
subtree.

```
ValueError: pytree structure error: different types at key path
    tree_map tree['__RKG']
prefix pytree: <class 'NoneType'>   full pytree: <class 'pcx.core._random.RandomKeyGenerator'>
```

Bisected: works on 0.4.33, fails on 0.4.34, 0.4.35, 0.5.3, 0.6.2, 0.7.2, 0.11.0.
Every tutorial notebook uses `pxf.vmap`, so **none of them run on a modern jax**.
`pyproject.toml` declared `jax = "^0.4.33"`, which permits newer, but the
committed `poetry.lock` pinned exactly 0.4.33 — so the authors' environments
worked while every fresh install was broken.

*Tests:* five in `tests/functional/test_transforms.py`, all named `test_vmap_*`
*Source:* `pcx/functional/_transform.py`, `Vmap._t`

### 5. A raised exception inside a transform poisons the global RNG

`_BaseTransform` swaps `RKG.key` for the traced key and restores it with a plain
statement rather than `try/finally`. If the transformed function raises, the
tracer stays installed in module-level state.

`pcx.RKG` is the default argument of every layer and Vode constructor, so **every
subsequent random draw in the process fails** with `UnexpectedTracerError` —
including in unrelated code that never used a transform. Verified end to end: one
failed `pxf.jit` call is enough.

*Tests:* `tests/functional/test_transforms.py::test_global_rkg_holds_a_concrete_array_after_a_transformed_function_raises`,
`tests/functional/test_flow.py::test_global_rkg_holds_a_concrete_array_after_a_flow_function_raises`
*Source:* `pcx/functional/_transform.py`, `_BaseTransform.__init__._map_fn._wrap_fn`

### 6. `value_and_grad` writes back to positionally-passed params

The library's central protocol is that positional arguments are pure jax values
and are not tracked, while keyword arguments are. `ValueAndGrad._t` forwards
`*args` straight into `jax.value_and_grad`, which leaves non-differentiated
positional arguments as the exact objects passed in — so mutating one inside the
function mutates the caller's object.

Worse, when the mutation involves the differentiated value, the caller's
positional `Param` is left holding a live autodiff tracer. Nothing raises at the
time; it detonates later as an unrelated `UnexpectedTracerError`.

`Jit` gets this right, so identical user code behaves differently depending on
which transform wraps it.

*Tests:* `tests/functional/test_transforms.py::test_value_and_grad_does_not_write_back_a_param_passed_positionally`,
`::test_value_and_grad_does_not_leak_a_tracer_into_a_positional_param`
*Source:* `pcx/functional/_transform.py`, `ValueAndGrad._t`

---

## Medium — crashes on ordinary usage

### 7. A cleared `ParamDict` raises `TypeError` on every read

`clear_params` sets a `ParamDict`'s value to `None` — the normal between-steps
state produced by `pxu.step(..., clear_params=...)`. `__setitem__` guards against
that state, but `__contains__`, `__getitem__` and `.get()` do not:

- `"E" in cache` → `TypeError: argument of type 'NoneType' is not iterable`
- `cache["E"]` → `TypeError: 'NoneType' object is not subscriptable`
- `cache.get("E", default)` → `AttributeError: 'NoneType' object has no attribute 'get'`

An empty mapping should report "not present", and `.get(key, default)` exists
precisely to be safe on absent keys. The first statement of `Vode.energy` is
`"E" not in self.cache`, so **`Vode.energy()` raises after any cache clear**.

*Tests:* three in `tests/core/test_parameter.py` (`*_cleared_paramdict_*`),
`tests/numerics/test_vode.py::test_a_cleared_cache_reads_as_empty_rather_than_raising`
*Source:* `pcx/core/_parameter.py`, `ParamDict`

### 8. `EnergyModule.energy()` raises on a model with no Vodes

`functools.reduce` is called without an initial value, so a module with no
`EnergyModule` children raises `TypeError: reduce() of empty iterable with no
initial value` instead of returning 0. Hits any model built incrementally, and
any leaf module in a hierarchy.

*Test:* `tests/numerics/test_vode.py::test_energy_of_a_module_with_no_vodes_is_zero`
*Source:* `pcx/predictive_coding/_energy_module.py`, `energy`

### 9. `pxu.step` does not restore status and has no `try/finally`

After `with pxu.step(model, STATUS.INIT):` the model is left in `'init'`. The
status is only reset when a 2-tuple is passed, so the canonical single-status
form never restores. Any energy computed between that block and the next runs
under the wrong ruleset.

The block also lacks `try/finally`, so an exception in the body skips both the
cache clear and the status reset, leaving the model corrupted — and under pytest
the next test inherits it.

*Source:* `pcx/utils/_misc.py`, `step`

### 10. `save_params` / `load_params` corrupts models with shared parameters

Duplicate references are written as literal `None`, which numpy stores as a
`dtype=object` array; `load_params` then calls `np.load` with the default
`allow_pickle=False` and refuses it:

```
ValueError: Object arrays cannot be loaded when allow_pickle=False
```

Checkpoints of any model using `pxnn.shared` are unrecoverable. The
`is not None` guard in the loader is dead code — `np.load` yields a 0-d object
array, never `None` — so even with `allow_pickle=True` the shared parameter
would be assigned garbage. Writer and reader need to agree on a sentinel.

*Source:* `pcx/utils/_serialisation.py`

### 11. `zero_energy` ignores the node's shape

`zero_energy` returns `jnp.zeros((1,))` regardless of its input. A
`Vode(energy_fn=zero_energy)` whose value has shape `(3, 2)` then raises
`TypeError: cannot reshape array of shape (1,) (size 1) into shape (3, -1)`
inside `Vode.energy`, so unconstrained nodes are unusable at any batch size ≠ 1.

*Test:* `tests/numerics/test_energy.py::test_zero_energy_has_the_same_shape_as_the_node_value`
*Source:* `pcx/predictive_coding/_energy.py`

---

## Low — API and contract inconsistencies

### 12. `pcx.set` cannot be called on a plain value

`set` on a non-param executes `obj = set(x)`, calling itself with one argument:
`TypeError: set() missing 1 required positional argument: 'x'`. The helper exists
so call sites need not know whether they hold a param, and that is the one path
that cannot work.

*Test:* `tests/core/test_parameter.py::test_set_on_a_plain_value_returns_the_new_value`

### 13. `tree_inject` cannot inject plain values

It calls `.get()` on every element of `values`, contradicting both the documented
type ("input sequence of values to inject") and the default
`inject_fn = lambda n, v: n.set(v)`. Arrays coming straight out of a transform
cannot be injected: `AttributeError: 'ArrayImpl' object has no attribute 'get'`.

*Test:* `tests/core/test_tree.py::test_inject_accepts_plain_values`

### 14. `tree_inject` raises bare `StopIteration` when given too few values

Surplus values raise a clear `ValueError`; a shortfall raises `StopIteration`
from `next()` inside the traversal. Under PEP 479 that becomes an unrelated
`RuntimeError` inside a generator, and the params visited before exhaustion have
already been overwritten.

*Test:* `tests/core/test_tree.py::test_inject_strict_rejects_missing_values`

### 15. `value_and_grad` rejects an integer `argnums`

Typed `argnums: int | Sequence[int]`, matching jax, but the implementation
unpacks it: `TypeError: Value after * must be an iterable, not int`. Only the
undocumented `(0,)` spelling works.

*Test:* `tests/functional/test_transforms.py::test_value_and_grad_accepts_an_integer_argnums`

### 16. `pxf.switch` rejects a list of branches

Typed `Sequence`, and `jax.lax.switch` conventionally takes a list, but
`_make_tuple` wraps a list instead of expanding it, collapsing all branches into
one callable: `TypeError: object of type 'function' has no len()`. Only a tuple
works.

*Test:* `tests/functional/test_flow.py::test_switch_accepts_a_list_of_branches`

### 17. `rkg(1)` is not uniform with `rkg(n)`

`rkg(n)` yields `n` keys of shape `(n, 2)` for `n >= 2`, but `rkg(1)` returns a
bare key of shape `(2,)`. So `keys[0]` is a uint32 rather than a key, and
iterating yields the key's two integers. Any batched path hits this on a final
partial batch or a single-example debug run.

*Test:* `tests/core/test_random.py::test_batch_draw_is_uniform_in_n`

### 18. Module treedefs depend on attribute assignment order

Treedefs key on `__dict__` insertion order, so two instances of one class whose
constructor assigned attributes in different orders cannot be `tree_map`ed
together and force a jit recompile. A module is documented as flattened "as if it
were a dictionary", and jax dicts are deliberately key-order-insensitive.

*Test:* `tests/core/test_module.py::test_attribute_assignment_order_does_not_change_the_treedef`

### 19. `Vode.energy()` returns a different shape depending on how `u` was set

`Vode.__call__` is documented as equivalent to `vode.set("u", u).get("h")`, but it
also records the input shape, and `energy` uses that record to choose between a
per-sample vector and a scalar sum. The two documented-equivalent paths give
shape `(3,)` and `()` respectively. Totals agree; per-sample resolution is lost.

*Test:* `tests/numerics/test_vode.py::test_energy_is_the_same_whether_u_was_set_by_call_or_by_set`

### 20. The `Scan` docstring example cannot produce its documented output

The class docstring's example — the library's only executable illustration of the
transform — is annotated `# [0, 1, 3, 6, 10], None`. The body adds `x` twice per
step, and the first return element is the final carry rather than a per-step
sequence, so the actual result is `((20,), None)`. The documented output is
unreachable by any scan.

*Test:* `tests/functional/test_flow.py::test_scan_docstring_example_produces_its_documented_output`

---

## Investigated and *not* a defect

- **`~` negation in the mask DSL.** Reported as never negating. Did not
  reproduce: `~M(LayerParam)` selected a different set from `M(LayerParam)`, as
  intended.
- **`tree_apply` visiting an aliased param once per occurrence.** Documented with
  a worked example, and `clear_params` depends on it. Correct as specified.
- **`Param` being unhashable.** `__eq__` is elementwise, so unhashability matches
  numpy/jax array semantics rather than contradicting them.
