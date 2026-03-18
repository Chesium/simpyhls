from dataclasses import dataclass, field
from typing import Iterable, Optional


class PrimitiveRTLError(Exception):
    pass


@dataclass
class PrimitiveRTLSpec:
    name: str
    ports: tuple[str, ...]
    kind: Optional[str] = None
    result_port: Optional[str] = None
    latency: Optional[int] = None
    start_signal: Optional[str] = None
    done_signal: Optional[str] = None
    arg_signals: dict[str, str] = field(default_factory=dict)
    result_signal: Optional[str] = None
    comb_function: Optional[str] = None
    comment: Optional[str] = None

    def __post_init__(self) -> None:
        self.ports = tuple(self.ports)
        if self.kind is None:
            self.kind = "comb" if self.name.endswith("_comb") else "blocking"
        if self.kind not in {"comb", "blocking"}:
            raise PrimitiveRTLError(f"Primitive '{self.name}' has invalid kind '{self.kind}'")

        if self.latency is None:
            self.latency = 0 if self.kind == "comb" else 1
        if self.latency < 0:
            raise PrimitiveRTLError(f"Primitive '{self.name}' cannot use negative latency")
        if self.kind == "comb" and self.latency != 0:
            raise PrimitiveRTLError(
                f"Combinational primitive '{self.name}' must use latency 0, got {self.latency}"
            )

        if self.kind == "comb":
            self.comb_function = self.comb_function or self.name
            self.start_signal = None
            self.done_signal = None
            self.result_signal = None
        else:
            self.start_signal = self.start_signal or f"{self.name}_start"
            self.done_signal = self.done_signal or f"{self.name}_done"
            self.result_signal = (
                self.result_signal
                or (f"{self.name}_{self.result_port}" if self.result_port else None)
            )

        for port in self.ports:
            self.arg_signals.setdefault(port, f"{self.name}_{port}")


@dataclass
class PrimitiveRTLRegistry:
    specs: dict[str, PrimitiveRTLSpec]

    def __init__(self, specs: dict[str, PrimitiveRTLSpec] | Iterable[PrimitiveRTLSpec]):
        if isinstance(specs, dict):
            normalized = dict(specs)
        else:
            normalized = {spec.name: spec for spec in specs}

        for name, spec in normalized.items():
            if name != spec.name:
                raise PrimitiveRTLError(
                    f"Primitive registry key '{name}' does not match spec name '{spec.name}'"
                )
        self.specs = normalized

    def require(self, name: str) -> PrimitiveRTLSpec:
        try:
            return self.specs[name]
        except KeyError as exc:
            raise PrimitiveRTLError(f"Missing RTL primitive spec for '{name}'") from exc

    def used(self, names: Iterable[str]) -> list[PrimitiveRTLSpec]:
        return [self.require(name) for name in dict.fromkeys(names)]


__all__ = ["PrimitiveRTLError", "PrimitiveRTLRegistry", "PrimitiveRTLSpec"]
