from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .lir import FuncLIR
from .lir_to_verilog_model import RTLModuleConfig, lower_to_verilog_model
from .primitive_rtl import PrimitiveRTLRegistry


def _template_dir() -> Path:
    return Path(__file__).resolve().parent / "templates" / "verilog"


def build_jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_template_dir())),
        undefined=StrictUndefined,
        trim_blocks=False,
        lstrip_blocks=True,
    )
    env.filters["sv_width"] = format_width
    env.filters["indent_lines"] = indent_lines
    env.filters["comment_escape"] = comment_escape
    return env


def format_width(width: int) -> str:
    return "" if width <= 1 else f"[{width - 1}:0] "


def indent_lines(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())


def comment_escape(text: str) -> str:
    return text.replace("*/", "* /").replace("\n", " ")


def generate_verilog(
    func_lir: FuncLIR,
    primitive_registry: PrimitiveRTLRegistry | dict | list,
    module_config: Optional[RTLModuleConfig] = None,
    *,
    include_debug: bool = False,
):
    registry = (
        primitive_registry
        if isinstance(primitive_registry, PrimitiveRTLRegistry)
        else PrimitiveRTLRegistry(primitive_registry)
    )
    rtl_module = lower_to_verilog_model(func_lir, registry, module_config)
    rendered = build_jinja_env().get_template("module.j2").render(module=rtl_module)
    if include_debug:
        return rendered, rtl_module.debug
    return rendered


__all__ = ["RTLModuleConfig", "build_jinja_env", "generate_verilog"]
