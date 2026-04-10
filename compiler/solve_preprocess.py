from __future__ import annotations

from dataclasses import dataclass


GROUND_SENTINEL = 65535
KIND_R = 1
KIND_I = 2
KIND_V = 3
KIND_VCVS = 4
KIND_NPN = 5


class SolvePreprocessError(Exception):
    pass


@dataclass(frozen=True)
class PreprocessedElement:
    index: int
    kind: int
    n0: int
    n1: int
    n2: int
    n3: int
    aux: int
    v0: float
    v1: float
    v2: float


@dataclass(frozen=True)
class PreprocessedSolveNetlist:
    elem_count: int
    dim: int
    elements: tuple[PreprocessedElement, ...]


def _node_to_idx(node: int) -> int:
    return GROUND_SENTINEL if node == 0 else node - 1


def preprocess_simple_solve_netlist(
    base_node_n: int,
    netlist: list[tuple],
) -> PreprocessedSolveNetlist:
    if base_node_n < 0:
        raise SolvePreprocessError("base_node_n must be non-negative")

    sorted_netlist = sorted(netlist, key=lambda item: item[0])
    seen_indices: set[int] = set()
    next_aux = base_node_n
    lowered: list[PreprocessedElement] = []

    for spec in sorted_netlist:
        if len(spec) < 5:
            raise SolvePreprocessError(
                f"Expected at least (index, type, ...) tuple, got {spec!r}"
            )

        index = spec[0]
        kind_name = spec[1]
        if index in seen_indices:
            raise SolvePreprocessError(f"Duplicate netlist index {index}")
        seen_indices.add(index)

        if kind_name == "R":
            if len(spec) != 5:
                raise SolvePreprocessError(
                    f"Resistor tuple must be (index, 'R', node1, node2, value), got {spec!r}"
                )
            _, _, node1, node2, value = spec
            lowered.append(
                PreprocessedElement(
                    index=index,
                    kind=KIND_R,
                    n0=_node_to_idx(node1),
                    n1=_node_to_idx(node2),
                    n2=GROUND_SENTINEL,
                    n3=GROUND_SENTINEL,
                    aux=GROUND_SENTINEL,
                    v0=1.0 / value,
                    v1=0.0,
                    v2=1.0,
                )
            )
            continue

        if kind_name == "I":
            if len(spec) != 5:
                raise SolvePreprocessError(
                    f"Current-source tuple must be (index, 'I', node1, node2, value), got {spec!r}"
                )
            _, _, node1, node2, value = spec
            lowered.append(
                PreprocessedElement(
                    index=index,
                    kind=KIND_I,
                    n0=_node_to_idx(node1),
                    n1=_node_to_idx(node2),
                    n2=GROUND_SENTINEL,
                    n3=GROUND_SENTINEL,
                    aux=GROUND_SENTINEL,
                    v0=float(value),
                    v1=0.0,
                    v2=1.0,
                )
            )
            continue

        if kind_name == "V":
            if len(spec) != 5:
                raise SolvePreprocessError(
                    f"Voltage-source tuple must be (index, 'V', node1, node2, value), got {spec!r}"
                )
            _, _, node1, node2, value = spec
            lowered.append(
                PreprocessedElement(
                    index=index,
                    kind=KIND_V,
                    n0=_node_to_idx(node1),
                    n1=_node_to_idx(node2),
                    n2=GROUND_SENTINEL,
                    n3=GROUND_SENTINEL,
                    aux=next_aux,
                    v0=float(value),
                    v1=0.0,
                    v2=1.0,
                )
            )
            next_aux += 1
            continue

        if kind_name == "E":
            if len(spec) != 7:
                raise SolvePreprocessError(
                    f"VCVS tuple must be (index, 'E', ctrl_p, ctrl_q, out_i, out_j, gain), got {spec!r}"
                )
            _, _, ctrl_p, ctrl_q, out_i, out_j, gain = spec
            lowered.append(
                PreprocessedElement(
                    index=index,
                    kind=KIND_VCVS,
                    n0=_node_to_idx(ctrl_p),
                    n1=_node_to_idx(ctrl_q),
                    n2=_node_to_idx(out_i),
                    n3=_node_to_idx(out_j),
                    aux=next_aux,
                    v0=float(gain),
                    v1=0.0,
                    v2=1.0,
                )
            )
            next_aux += 1
            continue

        if kind_name == "Q":
            if len(spec) == 6:
                _, _, base, collector, emitter, beta = spec
                vbe_drop = 0.7
            elif len(spec) == 7:
                _, _, base, collector, emitter, beta, vbe_drop = spec
            else:
                raise SolvePreprocessError(
                    f"NPN tuple must be (index, 'Q', base, collector, emitter, beta[, vbe_drop]), got {spec!r}"
                )
            lowered.append(
                PreprocessedElement(
                    index=index,
                    kind=KIND_NPN,
                    n0=_node_to_idx(base),
                    n1=_node_to_idx(collector),
                    n2=_node_to_idx(emitter),
                    n3=GROUND_SENTINEL,
                    aux=next_aux,
                    v0=float(beta),
                    v1=float(vbe_drop),
                    v2=1.0,
                )
            )
            next_aux += 1
            continue

        raise SolvePreprocessError(
            f"Unsupported simple-solver element type '{kind_name}'. "
            "Supported types are 'R', 'I', 'V', 'E', and 'Q'."
        )

    return PreprocessedSolveNetlist(
        elem_count=len(lowered),
        dim=next_aux,
        elements=tuple(lowered),
    )


__all__ = [
    "GROUND_SENTINEL",
    "KIND_I",
    "KIND_NPN",
    "KIND_R",
    "KIND_V",
    "KIND_VCVS",
    "PreprocessedElement",
    "PreprocessedSolveNetlist",
    "SolvePreprocessError",
    "preprocess_simple_solve_netlist",
]
