"""Find names that are bound ONLY inside a conditional branch but LOADED unconditionally later.

Why this exists: `oft_predict.py` had `import hashlib` inside `if is_frozen and not
args.allow_anchor_change:`, and the contract dump at the end of `main()` used `hashlib.md5`
unconditionally. On the delivery path (frozen val anchors) the branch runs and the name exists;
on stage 5's train anchors it does not, so the run would NameError *after* 7475 anchors of GPU
work. `py_compile` is blind to it (it is a runtime name error), and a unit test that injects the
name into a namespace is blind to it too -- the injection IS the bug being masked.

Deliberately conservative: reports only names bound exclusively under `if`/`for`/`while`/`try`
and later loaded at the function's own statement level, where "will it run" is not in question.
Import-time globals, builtins, args, and comprehension/except/with targets are all resolved first.
"""
import argparse
import ast
import builtins
import pathlib
import sys


# Comprehensions have their OWN scope in py3, so their targets never leak to the enclosing
# function. Omitting them made the checker flag a genexp's `b` as a function-level hazard.
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
          ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _iter_scope(node):
    """Yield `node` and its descendants IN THE SAME SCOPE. A nested def/class/lambda is yielded
    (it binds its own name here) but never entered.

    `ast.walk` cannot express this, and reaching for it is what made the first two versions of this
    checker no-ops -- both passed `py_compile`, ran clean, and proved nothing:
      v1: `ast.walk(module_FunctionDef)` pulled every function-LOCAL binding into the module-globals
          set, so every name looked already-safe.
      v2: added a `skip_self` flag, which controlled whether the def node was YIELDED but not
          whether its body was ENTERED -- so the root function's locals still leaked. The fix is
          structural (defs are leaves to their enclosing scope), not a flag.
    """
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, SCOPES):
            yield child
            continue
        yield from _iter_scope(child)


def _bound_names(node):
    """Every name this statement binds in the scope that CONTAINS it."""
    # A def/class statement binds exactly one name in its enclosing scope: its own.
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    out = set()
    for n in _iter_scope(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
    return out


def _loaded_names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return set()               # its body is a different scope, resolved when we check that def
    return {n.id for n in _iter_scope(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def check_function(fn, globals_):
    """Return [(lineno, name)] for unconditional loads of conditionally-bound names."""
    safe = set(globals_) | set(dir(builtins))
    for a in fn.args.args + fn.args.kwonlyargs + ([fn.args.vararg] if fn.args.vararg else []):
        if a:
            safe.add(a.arg)
    conditional = set()
    problems = []
    CONDITIONAL = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith)
    for stmt in fn.body:                       # statement level of the function only
        # A statement that binds a name itself cannot be tripped by it (`for x in ...:` using x in
        # its own body). Without this the checker flags its own `for lineno, name in hits:` loop.
        self_bound = _bound_names(stmt)
        for name in sorted(_loaded_names(stmt) - self_bound):
            if name in safe:
                continue
            if name in conditional:
                problems.append((stmt.lineno, name))
        if isinstance(stmt, CONDITIONAL):
            bound = _bound_names(stmt)
            # An if/else where BOTH arms bind a name DOES guarantee it (this is how the real
            # `pred = ...` in each arm of load_leg() is written). Take the intersection over all
            # arms as guaranteed; only the rest is conditional. `elif` nests as an If in orelse,
            # so recursion handles chains -- but a bare `if` with no `else` guarantees nothing.
            guaranteed = set()
            if isinstance(stmt, ast.If) and stmt.orelse:
                guaranteed = _guaranteed_by_if(stmt)
            safe |= guaranteed
            conditional |= bound - safe
            conditional -= guaranteed
        else:
            newly = _bound_names(stmt)
            safe |= newly
            conditional -= newly               # an unconditional rebind cures it
    return problems


def _guaranteed_by_if(node):
    """Names bound on EVERY path through an if/elif/else chain (empty if any arm is missing)."""
    if not node.orelse:
        return set()

    def arm(body):
        out = set()
        for s in body:
            if isinstance(s, ast.If):
                out |= _guaranteed_by_if(s)
            elif not isinstance(s, (ast.For, ast.While, ast.Try, ast.With)):
                out |= _bound_names(s)
        # a body that always raises/exits cannot fall through, so it constrains nothing
        if any(isinstance(s, (ast.Raise, ast.Return, ast.Continue, ast.Break)) for s in body):
            return None
        return out

    a, b = arm(node.body), arm(node.orelse)
    if a is None:
        return b or set()
    if b is None:
        return a
    return a & b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--function", default=None, help="limit to one function (default: all)")
    a = p.parse_args()
    tree = ast.parse(pathlib.Path(a.path).read_text())
    globals_ = set()
    for stmt in tree.body:
        if not isinstance(stmt, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
            globals_ |= _bound_names(stmt)
    hits = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if a.function and node.name != a.function:
                continue
            for lineno, name in check_function(node, globals_):
                hits.append((node.name, lineno, name))
    for fname, lineno, name in hits:
        print(f"{a.path}:{lineno}: {fname}() loads '{name}', which is only bound inside a "
              f"conditional branch -> NameError on the paths that skip it")
    print(f"{'FAIL' if hits else 'OK  '}  {a.path}  ({len(hits)} conditional-binding hazard(s))")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
