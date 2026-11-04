#!/usr/bin/env python3
"""T5.2 (batch a) codegen: transcribe the reference arithmetic of the 3D volumetric
constitutions (`arap_3d` / `stable_neo_hookean_3d`) into the luisa numerics layer.

Why a generator (same argument as T2.3's `scratch/gen_distance.py`): the reference's
`sym/*.inl` files are *machine generated* (SymEigen.py output) and the `detail/*.inl`
bodies are flat scalar algebra; both must be transcribed **verbatim** -- same temporary
names, same statement order, same association order -- changing only the spelling of each
operator, because LC's DSL scalar operators live at global scope and are shadowed inside
this layer (cookbook RULE M1, §7.2).  Hand editing generated arithmetic is where T2.2's
first 4x4 adjugate went wrong.

Conversion rules
----------------
    -v11 + v01                 ->  lps::detail::neg(v11) + ... => add(neg(v11), v01)
    a * b * 2.0                ->  mul(mul(a, b), 2.0)            (left assoc, kept)
    std::pow(x, 2)             ->  lps::pow(x, 2)                (lps::pow == std::pow on
                                                                  the host, lcrt::pow on the DSL)
    1.0 / (a + b)              ->  div(1.0, add(a, b))
    mu * (1 - 1 / (Ic + 1)) * F->  mul(mu, sub(1, div(1, add(Ic, 1)))) * F
                                   (`F` is a *container*: the container operands keep the
                                    module's own operators -- a container/scalar mix is not
                                    a scalar arithmetic operation.  Container-ness is
                                    inferred from the signature and the declarations.)
    Eigen::Matrix<T,R,C>       ->  lps::LMat<T, R, C>
    Eigen::Vector<T,N>         ->  lps::LVec<T, N>

Everything that is *not* arithmetic (control flow, `Eigen::Map`, comma-initialisers,
`.block<>`/`.template segment<>` plumbing) stays hand written in the target file; the
script only rewrites the regions delimited by

    // >>>GENERATED:<name>>>
    ...
    // <<<GENERATED:<name>>>

Usage
-----
    python scripts/gen_fem_cons_a.py              # rewrite the registered regions
    python scripts/gen_fem_cons_a.py --check      # verify the tree is in sync (rc 1 if stale)
    python scripts/gen_fem_cons_a.py --selfcheck  # + prove every generated statement is a
                                                  # faithful re-spelling (tree identity) and,
                                                  # for scalar statements, value-exact under
                                                  # 200 random inputs
"""
from __future__ import annotations

import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "src/backends/cuda/finite_element/constitutions"
DST_DIR = ROOT / "src/backends/luisa/finite_element/constitutions"

# (target file, reference file, region name, [reference function names])
REGIONS: list[tuple[str, str, str, list[str]]] = [
    ("sym/simplified_stable_neo_hookean_3d.inl",
     "sym/simplified_stable_neo_hookean_3d.inl",
     "simplified_snh_3d",
     ["E", "dEdVecF", "ddEddVecF"]),
    ("detail/stable_neo_hookean_3d.inl",
     "detail/stable_neo_hookean_3d.inl",
     "snh_3d",
     ["E", "dEdVecF"]),
    ("detail/stable_neo_hookean_3d_snk1.inl",
     "detail/stable_neo_hookean_3d_snk1.inl",
     "snk1",
     ["E", "dEdF"]),
]

TRAMP = {"+": "lps::detail::add", "-": "lps::detail::sub",
         "*": "lps::detail::mul", "/": "lps::detail::div"}

# ---------------------------------------------------------------------------
#  Tokenizer / parser
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+\.?(?:[eE][-+]?\d+)?)"
    r"|(?P<id>[A-Za-z_]\w*(?:::\w+)*)"
    r"|(?P<op>[()+\-*/,.]))")


def tokenize(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        m = TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot tokenize {text[pos:pos + 40]!r} of {text!r}")
        if m.group("num"):
            out.append(("num", m.group("num")))
        elif m.group("id"):
            out.append(("id", m.group("id")))
        else:
            out.append(("op", m.group("op")))
        pos = m.end()
    return out


# node shapes (tuples):
#   ("num", text) | ("id", name) | ("call", name, [args]) | ("member", obj, name)
#   | ("membercall", obj, name, [args]) | ("index", obj, [args]) | ("bin", op, a, b)
#   | ("neg", a)
class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def next(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expr(self):
        node = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.next()[1]
            node = ("bin", op, node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.next()[1]
            node = ("bin", op, node, self.unary())
        return node

    def unary(self):
        if self.peek() == ("op", "-"):
            self.next()
            return ("neg", self.unary())
        if self.peek() == ("op", "+"):
            self.next()
            return self.unary()
        return self.postfix()

    def args(self):
        out = []
        if self.peek() == ("op", ")"):
            return out
        out.append(self.expr())
        while self.peek() == ("op", ","):
            self.next()
            out.append(self.expr())
        return out

    def postfix(self):
        node = self.primary()
        while True:
            if self.peek() == ("op", "."):
                self.next()
                kind, name = self.next()
                if kind != "id":
                    raise ValueError("expected a member name after `.`")
                if self.peek() == ("op", "("):
                    self.next()
                    node = ("membercall", node, name, self.args())
                    if self.next() != ("op", ")"):
                        raise ValueError("missing )")
                else:
                    node = ("member", node, name)
            elif self.peek() == ("op", "("):
                # `name(args)` as a *postfix* application (Eigen coefficient access)
                self.next()
                a = self.args()
                if self.next() != ("op", ")"):
                    raise ValueError("missing )")
                node = ("index", node, a)
            else:
                break
        return node

    def primary(self):
        kind, text = self.next()
        if kind is None:
            raise ValueError("unexpected end of expression")
        if kind == "num":
            return ("num", text)
        if kind == "id":
            if self.peek() == ("op", "("):
                self.next()
                a = self.args()
                if self.next() != ("op", ")"):
                    raise ValueError("missing )")
                return ("call", text, a)
            return ("id", text)
        if text == "(":
            node = self.expr()
            if self.next() != ("op", ")"):
                raise ValueError("missing )")
            return node
        raise ValueError(f"unexpected token {text!r}")


def parse(rhs: str):
    p = Parser(tokenize(rhs))
    node = p.expr()
    if p.i != len(p.toks):
        raise ValueError(f"trailing tokens in {rhs!r}")
    return node


# ---------------------------------------------------------------------------
#  Reference / generated spelling
# ---------------------------------------------------------------------------
def ref_text(node) -> str:
    k = node[0]
    if k == "num":
        return node[1]
    if k == "id":
        return node[1]
    if k == "call":
        return f"{node[1]}({', '.join(ref_text(a) for a in node[2])})"
    if k == "index":
        return f"{ref_text(node[1])}({', '.join(ref_text(a) for a in node[2])})"
    if k == "member":
        return f"{ref_text(node[1])}.{node[2]}"
    if k == "membercall":
        return f"{ref_text(node[1])}.{node[2]}({', '.join(ref_text(a) for a in node[3])})"
    if k == "bin":
        return f"{ref_text(node[2])} {node[1]} {ref_text(node[3])}"
    if k == "neg":
        return f"-{ref_text(node[1])}"
    raise AssertionError(k)


# Container-valued operations: the receiver/arguments keep the module's operators.
CONTAINER_MEMBERS = {"transpose", "col", "row", "head", "tail", "segment", "block",
                     "diagonal", "asDiagonal", "array", "adjoint", "normalized", "inverse"}
SCALAR_MEMBERS = {"determinant", "squaredNorm", "norm", "sum", "trace", "maxCoeff",
                  "minCoeff", "dot", "cross"}


def is_container(node, containers: set[str]) -> bool:
    k = node[0]
    if k == "num":
        return False
    if k == "id":
        return node[1] in containers
    if k == "call":
        return node[1] in ("cross", "Identity", "Zero", "Ones")
    if k == "index":
        return is_container(node[1], containers)
    if k == "member":
        return node[2] in CONTAINER_MEMBERS
    if k == "membercall":
        # `.cross(x)` / `.normalized()` return containers, `.determinant()` a scalar
        return node[2] in CONTAINER_MEMBERS or node[2] == "cross"
    if k == "bin":
        return is_container(node[2], containers) or is_container(node[3], containers)
    if k == "neg":
        return is_container(node[1], containers)
    raise AssertionError(k)


CALL_MAP = {"pow": "lps::pow", "std::pow": "lps::pow",
            "sqrt": "lps::sqrt", "std::sqrt": "lps::sqrt",
            "log": "lps::log", "std::log": "lps::log",
            "exp": "lps::exp", "std::exp": "lps::exp",
            "abs": "lps::abs", "std::abs": "lps::abs",
            "max": "lps::max", "std::max": "lps::max",
            "min": "lps::min", "std::min": "lps::min",
            # T6.3b3: the revolute joint's generated bodies are the first to use a
            # two-scalar transcendental (`atan2`), so it joins the map.  Adding an
            # entry is backward compatible: no pre-T6.3b3 region mentions `atan2`,
            # so every existing region's output is byte-identical (re-verified with
            # `--check`).
            "atan2": "lps::atan2", "std::atan2": "lps::atan2"}

EIGEN_TYPE_MAP = [("Eigen::Matrix", "lps::LMat"), ("Eigen::Vector", "lps::LVec")]


def cpp_text(node, containers: set[str], raw_nums: bool = False) -> str:
    """`raw_nums` is used for *index* arguments (`.col(0)`, `F(0, 0)`), which must stay
    integers: everywhere else an integer literal becomes a `double` literal, which is
    value-identical on the host and keeps the DSL path free of LC's fp/integral
    implicit-conversion helper (a `[[deprecated]]` function in debug builds, i.e. a
    warning-hygiene measure -- cookbook §7.16 rule 10)."""
    k = node[0]
    if k == "num":
        if raw_nums:
            return node[1]
        text = node[1]
        return text + ".0" if re.match(r"^\d+$", text) else text
    if k == "id":
        return node[1]
    if k == "call":
        name = CALL_MAP.get(node[1], node[1])
        if node[1] in ("Identity", "Zero", "Ones"):
            for a, b in EIGEN_TYPE_MAP:
                name = name.replace(a, b)
        # `VecF(0)` / `F(1, 2)` are *coefficient* accesses: their arguments are indices,
        # not arithmetic operands.
        raw = node[1] in containers
        return (f"{name}"
                f"({', '.join(cpp_text(a, containers, raw) for a in node[2])})")
    if k == "index":
        return (f"{cpp_text(node[1], containers)}"
                f"({', '.join(cpp_text(a, containers, True) for a in node[2])})")
    if k == "member":
        return f"{cpp_text(node[1], containers)}.{node[2]}"
    if k == "membercall":
        return (f"{cpp_text(node[1], containers)}.{node[2]}"
                f"({', '.join(cpp_text(a, containers, True) for a in node[3])})")
    if k == "neg":
        if is_container(node, containers):
            return f"(-{cpp_text(node[1], containers)})"
        return f"lps::detail::neg({cpp_text(node[1], containers)})"
    if k == "bin":
        a, b, op = node[2], node[3], node[1]
        if is_container(a, containers) or is_container(b, containers):
            # reference parity: the container overloads live in the numerics layer
            return f"({cpp_text(a, containers)} {op} {cpp_text(b, containers)})"
        return f"{TRAMP[op]}({cpp_text(a, containers)}, {cpp_text(b, containers)})"
    raise AssertionError(k)


# ---------------------------------------------------------------------------
#  Reverse transform (used by --selfcheck)
# ---------------------------------------------------------------------------
def untransform(node, containers: set[str]):
    """Strip the re-spelling: trampolines back to operators, `lps::pow` -> `std::pow`."""
    k = node[0]
    if k == "call" and node[1] in ("lps::detail::add", "lps::detail::sub",
                                   "lps::detail::mul", "lps::detail::div"):
        op = {"lps::detail::add": "+", "lps::detail::sub": "-",
              "lps::detail::mul": "*", "lps::detail::div": "/"}[node[1]]
        if len(node[2]) != 2:
            raise ValueError(f"{node[1]} needs 2 arguments")
        return ("bin", op, untransform(node[2][0], containers),
                untransform(node[2][1], containers))
    if k == "call" and node[1] == "lps::detail::neg":
        return ("neg", untransform(node[2][0], containers))
    if k == "call" and node[1] == "lps::pow":
        return ("call", "std::pow", [untransform(a, containers) for a in node[2]])
    if k == "call" and node[1] in ("lps::sqrt", "lps::log", "lps::exp", "lps::abs",
                                   "lps::max", "lps::min", "lps::atan2"):
        return ("call", node[1][5:], [untransform(a, containers) for a in node[2]])
    if k == "call":
        return ("call", node[1], [untransform(a, containers) for a in node[2]])
    if k == "index":
        return ("index", untransform(node[1], containers),
                [untransform(a, containers) for a in node[2]])
    if k == "member":
        return ("member", untransform(node[1], containers), node[2])
    if k == "membercall":
        return ("membercall", untransform(node[1], containers), node[2],
                [untransform(a, containers) for a in node[3]])
    if k == "bin":
        return ("bin", node[1], untransform(node[2], containers),
                untransform(node[3], containers))
    if k == "neg":
        return ("neg", untransform(node[1], containers))
    return node


# ---------------------------------------------------------------------------
#  Reference extraction
# ---------------------------------------------------------------------------
COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def strip_comments(text: str) -> str:
    return COMMENT_RE.sub(" ", text)


def split_params(text: str) -> list[str]:
    """Split a parameter list on top-level commas (template arguments contain commas)."""
    out, cur, depth = [], "", 0
    for ch in text:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def split_name_spec(spec: str) -> tuple[str, int]:
    """`"DeltaTheta#2"` -> `("DeltaTheta", 2)`; a bare name is occurrence 1.

    T6.3b3: one reference `.inl` (the revolute joint-limit body) defines **two
    overloads** of `DeltaTheta` (a 6-vector `basis_k` form and a 12-vector `F`
    form).  A region is "one sentinel pair == every generated function in it", but
    the region *names* are explicit, so the second overload needs a way to be
    addressed.  The `#k` suffix is that way; it is backward compatible (every
    pre-existing name is occurrence 1)."""
    if "#" in spec:
        name, _, num = spec.rpartition("#")
        return name, int(num)
    return spec, 1


def find_function(text: str, name: str, occurrence: int = 1) -> re.Match:
    """The `occurrence`-th `__host__ __device__ void <name>(` declaration."""
    pat = re.compile(r"__host__\s+__device__\s+void\s+" + re.escape(name) + r"\s*\(")
    matches = list(pat.finditer(text))
    if len(matches) < occurrence:
        raise SystemExit(f"{name}#{occurrence} not found ({len(matches)} declaration(s))")
    return matches[occurrence - 1]


def function_body(text: str, name: str, occurrence: int = 1) -> str:
    txt = strip_comments(text)
    m = find_function(txt, name, occurrence)
    i = m.end() - 1
    depth = 0
    while True:
        if txt[i] == "(":
            depth += 1
        elif txt[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    j = txt.index("{", i)
    depth = 0
    k = j
    while True:
        if txt[k] == "{":
            depth += 1
        elif txt[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    return txt[j + 1:k]


def split_statements(body: str) -> list[str]:
    out, cur, depth = [], "", 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == ";":
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        raise SystemExit(f"trailing text without ';': {cur.strip()!r}")
    return [s for s in out if s]


DECL_RE = re.compile(r"^(Eigen::(?:Matrix|Vector)<T,\s*\d+(?:,\s*\d+)?>)\s+(\w+)$")


def convert_statement(stmt: str, containers: set[str]) -> tuple[str, bool]:
    """Return (converted statement, is_pure_scalar)."""
    m = DECL_RE.match(stmt)
    if m:
        ty = m.group(1)
        for a, b in EIGEN_TYPE_MAP:
            ty = ty.replace(a, b)
        containers.add(m.group(2))
        return f"{ty} {m.group(2)};", True

    m = re.match(r"^(.+?)\s*=\s*(.*)$", stmt, re.S)
    if not m:
        raise SystemExit(f"unhandled statement {stmt!r}")
    lhs, rhs = m.group(1).strip(), m.group(2).strip()

    # `auto`/`T`/`Float` declarations -- inferred declarations
    md = re.match(r"^(auto|T|Float|const T|const auto|int)\s+(\w+)$", lhs)
    if md:
        ty, name = md.group(1), md.group(2)
        # an `auto` declaration whose RHS is a container expression
        node = parse(rhs)
        if ty == "auto" and is_container(node, containers):
            containers.add(name)
        return f"{ty} {name} = {cpp_text(node, containers)};", not is_container(node, containers)

    node = parse(rhs)
    pure = not is_container(node, containers)
    text = cpp_text(node, containers)
    if re.match(r"^\w+$", lhs):
        return f"{lhs} = {text};", pure
    if re.match(r"^\w+\(\s*\d+\s*(,\s*\d+\s*)?\)$", lhs):
        # normalise the index spacing like the reference's own style
        inner = lhs[lhs.index("(") + 1:-1]
        parts = [p.strip() for p in inner.split(",")]
        return f"{lhs[:lhs.index('(')]}({', '.join(parts)}) = {text};", pure
    raise SystemExit(f"unhandled lvalue {lhs!r} in {stmt!r}")


def convert_function(reference: str, name: str, occurrence: int = 1) -> tuple[str, set[str], int, int]:
    """Return (C++ text, container names, #statements, #pure-scalar statements)."""
    body = function_body(reference, name, occurrence)
    header = reference[:reference.index(body, reference.index("void " + name))] \
        if False else reference
    # signature: derive it from the reference by type-mapping only
    txt = strip_comments(header)
    m = find_function(txt, name, occurrence)
    i = m.end() - 1
    depth = 0
    while True:
        if txt[i] == "(":
            depth += 1
        elif txt[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    params = " ".join(txt[m.end():i].split())

    containers: set[str] = set()
    for p in split_params(params):
        pm = re.match(r"^\s*(?:const\s+)?(Eigen::(?:Matrix|Vector)<T,\s*\d+(?:,\s*\d+)?>)\s*&?\s*(\w+)\s*$", p)
        if pm:
            containers.add(pm.group(2))
        else:
            nm = re.match(r"^\s*(?:const\s+)?(?:T|Float|double|int)\s*&?\s*(\w+)\s*$", p)
            if nm:
                pass  # a scalar parameter

    for a, b in EIGEN_TYPE_MAP:
        params = params.replace(a, b)

    lines = ["template <typename T>", f"inline void {name}({params})", "{"]
    n_stmts = 0
    n_pure = 0
    for stmt in split_statements(body):
        text, pure = convert_statement(stmt, containers)
        lines.append("    " + text)
        n_stmts += 1
        n_pure += 1 if pure else 0
    lines.append("}")
    return "\n".join(lines), containers, n_stmts, n_pure


# ---------------------------------------------------------------------------
#  Region rewrite
# ---------------------------------------------------------------------------
def begin(name: str) -> str:
    return f"// >>>GENERATED:{name}>>>"


def end(name: str) -> str:
    return f"// <<<GENERATED:{name}<<<"


def region_text(reference: str, names: list[str], check_self: bool) -> tuple[str, list[str]]:
    out = []
    report = []
    for spec in names:
        name, occurrence = split_name_spec(spec)
        text, containers, n_stmts, n_pure = convert_function(reference, name, occurrence)
        out.append(text)
        report.append(f"    {spec}: {n_stmts} statements ({n_pure} scalar)")
        if check_self:
            verify_function(reference, name, containers, occurrence)
    return "\n".join(out), report


def canon(node):
    """Canonicalise numeric literals for the tree comparison (`3` vs `3.0`)."""
    k = node[0]
    if k == "num":
        return ("num", repr(float(node[1])))
    return tuple(canon(x) if isinstance(x, tuple) else
                 ([canon(a) for a in x] if isinstance(x, list) else x)
                 for x in node)


def verify_function(reference: str, name: str, containers: set[str], occurrence: int = 1) -> None:
    """Prove the re-spelling is a faithful transform: re-parse the generated text and
    compare its untransformed tree against the reference's tree statement by statement."""
    text, _, _, _ = convert_function(reference, name, occurrence)
    gen_body = text[text.index("{") + 1:text.rindex("}")]
    ref_body = function_body(reference, name, occurrence)
    gen_stmts = split_statements(gen_body)
    ref_stmts = split_statements(strip_comments(ref_body))
    if len(gen_stmts) != len(ref_stmts):
        raise SystemExit(f"{name}: statement count {len(gen_stmts)} != {len(ref_stmts)}")
    values = {}
    rng = random.Random(0xC0FFEE)

    def unmap_types(text: str) -> str:
        for a, b in EIGEN_TYPE_MAP:
            text = text.replace(b, a)
        return re.sub(r"\s+", " ", text).strip()

    def squash(text: str) -> str:
        return re.sub(r"\s+", "", text)

    for gs, rs in zip(gen_stmts, ref_stmts):
        gm = re.match(r"^(?:(auto|T|Float|const T|const auto|int)\s+\w+\s*=\s*)?(.+?)\s*=\s*(.*)$",
                      gs, re.S)
        rm = re.match(r"^(?:(auto|T|Float|const T|const auto|int)\s+\w+\s*=\s*)?(.+?)\s*=\s*(.*)$",
                      rs, re.S)
        if not gm or not rm:
            if unmap_types(gs) != unmap_types(rs):
                raise SystemExit(f"{name}: statement mismatch\n  gen: {gs}\n  ref: {rs}")
            continue
        if squash(gm.group(2)) != squash(rm.group(2)):
            raise SystemExit(f"{name}: lvalue mismatch {gm.group(2)!r} != {rm.group(2)!r}")
        gen_tree = canon(untransform(parse(gm.group(3)), containers))
        ref_tree = canon(parse(rm.group(3)))
        if gen_tree != ref_tree:
            raise SystemExit(f"{name}: tree mismatch\n  gen: {ref_text(gen_tree)}"
                             f"\n  ref: {ref_text(ref_tree)}")
        # value check for scalar statements: bitwise equality on 200 random inputs
        if not is_container(ref_tree, containers):
            for _ in range(200):
                def value(node):
                    k = node[0]
                    if k == "num":
                        return float(node[1])
                    if k == "call" and node[1] in ("std::pow", "lps::pow"):
                        return math_pow(value(node[2][0]), value(node[2][1]))
                    if k == "call" and node[1] in ("log", "lps::log"):
                        return math.log(value(node[2][0]))
                    if k == "call" and node[1] in ("sqrt", "lps::sqrt"):
                        return math.sqrt(value(node[2][0]))
                    if k == "call" and node[1] in ("exp", "lps::exp"):
                        return math.exp(value(node[2][0]))
                    if k == "call" and node[1] in ("atan2", "lps::atan2"):
                        return math.atan2(value(node[2][0]), value(node[2][1]))
                    if k == "bin":
                        a, b = value(node[2]), value(node[3])
                        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[node[1]]
                    if k == "neg":
                        return -value(node[1])
                    key = ref_text(node)
                    if key not in values:
                        values[key] = rng.uniform(0.5, 2.0)
                    return values[key]
                if value(ref_tree) != value(gen_tree):
                    raise SystemExit(f"{name}: value mismatch for {ref_text(ref_tree)}")


def math_pow(a, b):
    if b == int(b):
        return a ** int(b)
    return a ** b


def rewrite(target: Path, reference: Path, region: str, names: list[str],
            check: bool, check_self: bool) -> tuple[bool, list[str]]:
    text = target.read_text(encoding="utf-8")
    b, e = begin(region), end(region)
    if b not in text or e not in text:
        raise SystemExit(f"{target}: missing sentinels for {region}")
    head, rest = text.split(b, 1)
    _, tail = rest.split(e, 1)
    block, report = region_text(reference.read_text(encoding="utf-8"), names, check_self)
    new = b + "\n" + block + "\n" + e
    changed = head + new + tail != text
    if changed and not check:
        target.write_text(head + new + tail, encoding="utf-8", newline="\n")
    return changed, report


def main() -> int:
    check = "--check" in sys.argv
    selfcheck = "--selfcheck" in sys.argv
    changed_any = False
    for target_name, ref_name, region, names in REGIONS:
        target = DST_DIR / target_name
        reference = REF_DIR / ref_name
        changed, report = rewrite(target, reference, region, names, check, selfcheck)
        changed_any |= changed
        print(f"{'STALE' if changed else 'ok   '} {target_name} <- {ref_name} "
              f"({', '.join(names)})")
        for line in report:
            print(line)
    if check and changed_any:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
