# BPC C++ Label Pricing Memory

Date: 2026-05-23

This note records the agreed implementation logic for the C++ label-pricing
subproblem in `BPC_label_cpp`. It is meant as a project memory for later coding,
testing, and paper-writing work.

## Scope

The relevant production implementation is the C++ solver under:

- `BPC_label_cpp/include/bpc_label/LabelPricing.hpp`
- `BPC_label_cpp/src/LabelPricing.cpp`
- `BPC_label_cpp/src/WagonPricing.cpp`
- `BPC_label_cpp/src/BpcSolver.cpp`
- `BPC_label_cpp/src/solver_main.cpp`

The active command-line modes are:

```text
--profile-generator-mode ex|d2|hyb
```

Old names and old mechanisms such as `gr`, `OrderOnly`,
`order-dominance-scope`, `rho_h`, `required_hits`,
`labels_pruned_by_order`, and `placements_skipped_by_order` are not part of the
current C++ production path.

## Current Algorithm Meanings

### EX

`ex` is the baseline exact label-pricing mode.

For each automobile type extension, the code enumerates all feasible component
placement consumptions for each quantity choice. Each child residual profile is
materialized, then D1 can prune dominated labels.

This mode is the reference path for exactness checks.

### D1

D1 is the always-valid residual-profile dominance rule.

At the same type-extension stage, label `A` dominates label `B` only when:

- `reduced_cost(A) <= reduced_cost(B)`;
- `quantity(A) <= quantity(B)` componentwise;
- `residual(A) >= residual(B)` over every nested checking region;
- the two labels are not identical.

D1 is a profile-level dominance rule. It does not depend on geometric ordering.
It remains valid for arbitrary component-dependent loading lengths and arbitrary
height-feasibility patterns.

D1 may still prune labels after D2/HYB has reduced placement generation, because
D2 keeps one canonical placement per quantity vector but D1 can compare labels
across different quantity vectors or different parent states.

### D2

`d2` is the certified ordered-greedy generation mode.

D2 is exact only when the whole automobile type set can be certified before the
main label loop. The current implementation builds an ordered chain using the
structural sufficient conditions in `type_less_restrictive(...)`: length
ordering, placement-set dominance, the ordered exchange certificate, and the
profile replacement certificate. If the whole type set is not certified, `d2`
falls back to `ex`.

When certified, D2 does not enumerate component-placement vectors. For each
quantity choice of the current type, it inserts vehicles greedily into the first
feasible component under the fixed ordered component list. Thus each quantity
choice produces at most one child residual profile.

The statistic `labels_avoided_by_d2` counts the placement labels avoided by this
canonical D2 generation.

Important exactness boundary:

- "Types are length/height ordered" alone is not enough.
- The structural ordered-greedy certificate is required.
- If the certificate fails, D2 must not silently act as greedy-only; it must
  fall back to exact enumeration.

### HYB / D3

`hyb` is the current D3+D1 production mode.

The implementation chooses a globally certified ordered chain. Types not in that
chain are treated as conflict types. The search order is:

1. conflict types first;
2. ordered certified types afterward.

For the conflict prefix, the code uses exact placement enumeration and D1.
After the iteration reaches the ordered suffix, the code uses D2-style greedy
generation for the certified ordered types, and still applies D1 after child
generation.

The ordered suffix is decided globally before the main label loop. The code
should not enumerate conflict residual states before the label loop, and should
not introduce local per-residual D2 certificates unless the paper logic is
changed deliberately.

## Residual Update Rule

Residual resources are nested checking-region lengths.

When one vehicle of type `i` is placed in component `h`, every region containing
`h` is reduced by the component-dependent adjusted loading length:

```text
l_i^h + Delta
```

This is important. The update must not use only the base type length when
component-dependent loading lengths or perturbations are present.

## Code-Level Landmarks

In `BPC_label_cpp/src/LabelPricing.cpp`:

- `parse_generator_mode(...)` recognizes `ex`, `d2`, and `hyb`.
- `placement_consumptions(...)` is the exact component-placement enumeration
  path.
- `greedy_child_for_residual(...)` is the D2 canonical child-generation path.
- `type_less_restrictive(...)` checks the structural D2 certificate for each
  ordered type pair.
- `longest_order_compatible_chain(...)` is used by D2 and HYB/D3 to choose the
  certified ordered set before the main label loop.
- `article_search_order(...)` builds the type order: exact order for EX,
  all-certified order for D2, conflict-prefix plus ordered-suffix for HYB.
- `apply_global_d1_dominance(...)` and `apply_local_d1_dominance(...)` implement
  D1 pruning.

Current label statistics:

- `labels_generated_raw`
- `labels_feasible`
- `labels_pruned_by_bound`
- `labels_pruned_by_dominance`
- `labels_after_dominance`
- `labels_pruned_total`
- `labels_avoided_by_d2`

## Cleanup Decisions Already Made

The C++ production code and C++ batch entry scripts were cleaned to use
`ex|d2|hyb` consistently.

Updated C++-calling scripts include:

- `scripts/benchmark_cpp_core_methods.py`
- `scripts/compare_cpp_bpc.py`
- `scripts/run_real_case_compartment_label.py`
- `scripts/run_batch_experiments.ps1`
- `scripts/run_cpp_core_batch_experiments.ps1`
- `src/utility/batch_experiments.py`
- `src/utility/cpp_core_batch_experiments.py`
- `src/experiments/compare_bpc_pricing_methods.py`

The tracked old Windows binary `BPC_label_cpp/bpc_label_solver.exe` was removed
from the repository state, and `*.exe` was added to
`BPC_label_cpp/.gitignore`.

Some old Python research prototype scripts still mention GR-like experimental
logic. They were intentionally not deleted because they are outside the C++
production path and may preserve old counterexample or manuscript exploration
work.

## Validation Snapshot

After the cleanup, the following checks passed:

```text
python3 -m py_compile scripts/benchmark_cpp_core_methods.py \
  scripts/compare_cpp_bpc.py \
  scripts/run_real_case_compartment_label.py \
  src/utility/cpp_core_batch_experiments.py \
  src/utility/batch_experiments.py \
  src/experiments/compare_bpc_pricing_methods.py

make bpc_label_smoke bpc_label_solver

./bpc_label_smoke

git diff --check
```

Small regression results:

```text
ordered_m5, compartment:
  EX  objective = -246000, generated_subpatterns = 1715
  D2  objective = -246000, generated_subpatterns = 1715
  HYB objective = -246000, generated_subpatterns = 1715

m5c5, compartment:
  EX  objective = -238342, generated_subpatterns = 1580
  D2  objective = -238342, generated_subpatterns = 1580
  HYB objective = -238342, generated_subpatterns = 1580
```

These tests support exactness for the cleaned modes on the checked instances.

## Important Collaboration Rule

This is paper code. Do not silently change the theoretical logic of D1, D2, or
D3. If a proposed implementation detail changes the paper-level algorithm,
state the issue first and ask before editing.
