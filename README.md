# simpyhls

**simpyhls** is an experimental Python-like DSL to Verilog generator for simple, control-oriented hardware modules.

The project is aimed at a common FPGA prototyping workflow:

1. write a sequential Python reference kernel
2. constrain it to a hardware-oriented DSL
3. lower it into a structured IR
4. generate readable FSM-style Verilog

simpyhls is intentionally narrow. It does **not** try to be a full optimizing HLS framework. The current focus is:

- single-thread control flow
- explicit blocking operations
- readable generated RTL
- clear source-to-IR-to-RTL provenance
- easy semantic testing against Python models

## Current workflow

The supported backend flow is now:

**DSL source -> Python AST -> HIR -> LIR -> RTL view-model -> Jinja-rendered Verilog**

The easiest public entry point is the workflow API in [compiler/workflow.py](c:/C/EE2026/simpyhls/compiler/workflow.py):

- `compile_source(source, primitive_registry, module_config=None)`
- `compile_file(path, primitive_registry, module_config=None)`
- `format_codegen_report(report)`
- `write_compilation_outputs(artifact, verilog_path, report_path=None)`

This is the recommended way to use the compiler because it:

- parses and lowers only once
- reuses the same RTL view-model for reporting and Verilog rendering
- returns HIR, LIR, rendered Verilog, and a structured report in one object

## DSL shape

The current DSL is intentionally restrictive. The frontend expects:

- a single top-level function
- typed-by-name locals such as `u8_i`, `f32_x`, `par_n`
- local declarations initialized near the top of the function
- structured `if`, `for`, and `while`
- primitive calls with **named arguments**

Primitive naming convention:

- calls ending in `_comb` are treated as combinational
- all other primitive calls are treated as blocking hardware operations

Typical example:

```python
def lu_core(par_n):
    f32_1 = 0
    f32_2 = 0
    u8_i = 0
    u8_j = 0
    for u8_j in range(par_n):
        for u8_i in range(par_n):
            f32_1 = fetch_A(i=u8_i, j=u8_j)
            f32_1 = neg_comb(v=f32_1)
            store_LU(i=u8_i, j=u8_j, v=f32_1)
```

## Installation

The project currently targets Python `>=3.14` and uses Jinja for Verilog rendering.

If you are using `uv`:

```powershell
uv sync
```

If you are using an existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

## End-to-end code generation

### 1. Define primitive RTL metadata

Primitive metadata tells the Verilog backend how to expose each primitive on the generated module interface.

Use [PrimitiveRTLSpec](c:/C/EE2026/simpyhls/compiler/primitive_rtl.py) and [PrimitiveRTLRegistry](c:/C/EE2026/simpyhls/compiler/primitive_rtl.py):

```python
from compiler.primitive_rtl import PrimitiveRTLRegistry, PrimitiveRTLSpec

registry = PrimitiveRTLRegistry(
    [
        PrimitiveRTLSpec(name="neg_comb", ports=("v",)),
        PrimitiveRTLSpec(name="fetch_A", ports=("i", "j"), result_port="result", latency=2),
        PrimitiveRTLSpec(name="fetch_LU", ports=("i", "j"), result_port="result", latency=2),
        PrimitiveRTLSpec(name="fma", ports=("a", "b", "c"), result_port="result", latency=3),
        PrimitiveRTLSpec(name="div", ports=("a", "b"), result_port="result", latency=4),
        PrimitiveRTLSpec(name="store_LU", ports=("i", "j", "v"), latency=1),
    ]
)
```

Rules:

- `kind` defaults to `"comb"` for names ending in `_comb`, otherwise `"blocking"`
- blocking primitives default to latency `1` unless overridden
- combinational primitives must use latency `0`
- blocking primitives automatically get default handshake signal names like `fetch_A_start`, `fetch_A_done`, `fetch_A_result`

### 2. Compile a DSL file

```python
from pathlib import Path

from compiler.workflow import compile_file, format_codegen_report, write_compilation_outputs

artifact = compile_file("examples/lu_core.dsl.py", registry)

print(artifact.verilog)
print()
print(format_codegen_report(artifact.report))

write_compilation_outputs(
    artifact,
    "build/lu_core.sv",
    "build/lu_core.report.txt",
)
```

`artifact` contains:

- `artifact.hir`
- `artifact.lir`
- `artifact.rtl_module`
- `artifact.verilog`
- `artifact.report`

### 3. Optional module configuration

You can override top-level signal names with [RTLModuleConfig](c:/C/EE2026/simpyhls/compiler/lir_to_verilog_model.py):

```python
from compiler.lir_to_verilog_model import RTLModuleConfig
from compiler.workflow import compile_file

config = RTLModuleConfig(
    module_name="lu_core_top",
    clock_signal="clk",
    reset_signal="rst_n",
    start_signal="start",
    busy_signal="busy",
    done_signal="done",
    return_signal="ret_val",
    emit_comments=True,
)

artifact = compile_file("examples/lu_core.dsl.py", registry, config)
```

## What the generated Verilog looks like

The backend currently generates a readability-first 2-process FSM:

- one module per DSL function
- `state` and `next_state`
- one `always_ff` block for state/register updates
- one `always_comb` block for defaults, control, and next-state logic
- explicit primitive handshake ports for blocking operations
- source/LIR comments above each generated state

Comments are sourced from:

- DSL line numbers and source snippets
- LIR block labels
- blocking primitive latency metadata in the module header

## Report generation

The report is intended to be cheap, stable, and easy to compute from the existing compiler pipeline.

The structured report type is [CodegenReport](c:/C/EE2026/simpyhls/compiler/workflow.py), and the pretty-printer is `format_codegen_report(report)`.

Current report fields include:

- function name
- module name
- control-flow counts: `if`, `for`, `while`
- primitive usage summary
- blocking call-site list with source lines and nesting context
- LIR block count
- RTL state count
- RTL wait-state count
- local and temporary register counts
- a conservative blocking-wait latency estimate

### Latency report semantics

The current latency report is intentionally conservative and simple:

- it counts **blocking primitive wait cycles only**
- it does **not** count all FSM control cycles
- it reports an exact bound only when the structure is statically obvious
- it falls back to a symbolic expression for runtime-dependent loops and branches

Examples:

- a fixed loop with two `load(latency=3)` iterations reports exact blocking wait latency `6`
- a loop over `range(par_n)` reports a symbolic form like `max(par_n, 0) * (...)`
- a `while` loop reports unknown upper bound

This makes the report useful for:

- spotting which primitives dominate latency
- seeing where blocking operations sit in nested loops
- understanding whether a latency number is exact or only symbolic

It is **not** yet a full cycle-accurate performance model of the generated RTL.

## Semantic testbench runtime

The project also includes a semantic testbench runtime for comparing:

- direct DSL execution in Python
- HIR interpretation
- LIR interpretation

Relevant modules:

- [compiler/sim_runtime.py](c:/C/EE2026/simpyhls/compiler/sim_runtime.py)
- [compiler/hir_sim.py](c:/C/EE2026/simpyhls/compiler/hir_sim.py)
- [compiler/lir_sim.py](c:/C/EE2026/simpyhls/compiler/lir_sim.py)

Users provide Python primitive models plus backing state, and the runtime deep-copies the initial state so all runs start from the same data.

Example outline:

```python
from compiler.sim_runtime import PrimitiveModel, SimulationHarness, run_all

def fetch_A(tb, *, i, j):
    return tb.state["A"][i][j]

def store_LU(tb, *, i, j, v):
    tb.state["LU"][i][j] = v

harness = SimulationHarness(
    params={"par_n": 4},
    primitives=[
        PrimitiveModel("fetch_A", fetch_A, latency=2),
        PrimitiveModel("store_LU", store_LU, latency=1),
    ],
    initial_state={
        "A": [[1.0, 2.0], [3.0, 4.0]],
        "LU": [[0.0, 0.0], [0.0, 0.0]],
    },
)

report = run_all(Path("examples/lu_core.dsl.py").read_text(), harness)
print(report.ok)
```

This is the recommended way to validate semantic equivalence before trusting a new lowering step or backend change.

## Recommended user workflow

For a new kernel:

1. Write the DSL function in `examples/` or your own source file.
2. Define primitive Python models and validate semantics with `run_all(...)`.
3. Define matching `PrimitiveRTLSpec` entries for the same primitives.
4. Call `compile_file(...)` once and inspect:
   - `artifact.hir`
   - `artifact.lir`
   - `artifact.verilog`
   - `artifact.report`
5. Write `artifact.verilog` and `format_codegen_report(artifact.report)` to disk.
6. Review the report before synthesis:
   - are the blocking primitives correct?
   - are loop/branch counts what you expect?
   - does the latency summary look plausible?
   - do the generated comments point back to the right DSL lines?

This is the current optimized workflow because the compile artifact is reused for all downstream inspection instead of rebuilding the backend multiple times.

## Debugging tips

Useful things to print during development:

```python
print(artifact.hir)
print()
print(artifact.lir)
print()
print(artifact.verilog)
print()
print(format_codegen_report(artifact.report))
```

Helpful checks:

- if the frontend rejects a kernel, start with [compiler/ast_checker.py](c:/C/EE2026/simpyhls/compiler/ast_checker.py)
- if control flow looks wrong, inspect [compiler/ast_to_hir.py](c:/C/EE2026/simpyhls/compiler/ast_to_hir.py) and [compiler/hir_to_lir.py](c:/C/EE2026/simpyhls/compiler/hir_to_lir.py)
- if interface ports or wait states look wrong, inspect [compiler/primitive_rtl.py](c:/C/EE2026/simpyhls/compiler/primitive_rtl.py) and [compiler/lir_to_verilog_model.py](c:/C/EE2026/simpyhls/compiler/lir_to_verilog_model.py)
- if rendered Verilog formatting looks wrong, inspect [compiler/verilog_codegen.py](c:/C/EE2026/simpyhls/compiler/verilog_codegen.py) and the templates under [compiler/templates/verilog](c:/C/EE2026/simpyhls/compiler/templates/verilog/module.j2)

## Validation

The repository now includes automated tests for:

- AST -> HIR lowering
- HIR pretty-printing
- HIR -> LIR lowering
- semantic equivalence across Python / HIR / LIR simulation
- LIR -> RTL view-model lowering
- Jinja-based Verilog generation
- the compile/report workflow
- the full `lu_core` example

Run the test suite with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Current limitations

The backend is still intentionally conservative.

Current limitations include:

- no aggressive scheduling or resource sharing
- no full cycle-accurate latency model in the report
- blocking handshake behavior is metadata-driven, not interface-inferred
- the DSL and generated RTL conventions are still evolving

That said, the full frontend-to-Verilog path is now implemented and test-covered, and `lu_core` can be lowered all the way to readable Verilog with an accompanying structural report.
