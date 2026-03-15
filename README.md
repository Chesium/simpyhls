# simpyhls

**simpyhls** is an experimental Python-like DSL to Verilog code generator for simple hardware control-oriented modules.

The project is motivated by a common workflow in FPGA prototyping: first writing a sequential Python reference implementation, then manually translating it into a one-thread FSM-style Verilog module with explicit `start / busy / done` handshaking. simpyhls aims to automate part of that process for a **restricted, hardware-oriented DSL** rather than act as a general-purpose HLS tool.

## Current idea

The intended flow is:

**Python-like DSL → Python AST → internal IR → Verilog code generation**

The DSL is designed to describe modules in a style close to “assembly with structured control flow”:

* a single top-level function
* integer-like top-level parameters
* local variables declared and initialized at the top
* explicit `for` / `while` / `if`
* explicit primitive calls for RAM access and long-latency operations
* named arguments for primitive calls to map clearly onto hardware ports

Primitive calls ending with `_comb` are intended to represent combinational operations, while other primitives are treated as blocking hardware operations that will later lower into explicit FSM sequencing.

## Design philosophy

simpyhls is **not** currently intended to compete with mainstream HLS frameworks. Instead, it focuses on a much narrower and more predictable problem:

* single-thread control flow
* explicit blocking operations
* readable generated RTL
* fast iteration from reference model to synthesizable Verilog

The goal is to make the generated hardware structure easy to understand and debug, even if it is not aggressively optimized.

## Status

This project is **still in active design and specification development**.

The DSL syntax, semantic rules, IR structure, primitive interface conventions, and Verilog generation templates are all still evolving. Current discussions have mainly focused on defining:

* the minimal DSL subset
* frontend checking rules using Python `ast`
* naming conventions for variables
* primitive call conventions
* a future lowering path into FSM-style Verilog

As a result, **the language and compiler architecture should be considered provisional**. Expect changes in syntax, restrictions, metadata handling, and generated RTL structure as the project matures.

## Near-term direction

Planned work includes:

* a frontend checker for the restricted DSL
* an internal IR for structured control flow and primitive calls
* lowering into FSM-oriented control steps
* Jinja-based Verilog code generation
* simple primitive metadata support for interface and latency description

simpyhls is best viewed, for now, as a developing research/prototyping tool for structured control-centric hardware generation.
