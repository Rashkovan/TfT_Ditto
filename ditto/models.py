from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AnnotatedVariable:
    name: str
    tag: str          # "prior" | "observed" | "approx"
    expr: str         # raw RHS expression string
    line: int         # source line number of the annotation comment


@dataclass
class VariableResult:
    name: str
    tag: str
    expr: str
    line: int
    samples: List[float] = field(default_factory=list)


@dataclass
class UploadResult:
    session_id: str
    filename: str
    version: int
    variables: List[VariableResult] = field(default_factory=list)
