"""
Guard test: GNCELoss (model/auxiliary_losses.py) and GNCELossAblation
(model/auxiliary_losses_ablation.py) MUST compute the same nce_loss on the
'roll' negative path — otherwise the Phase-B (main) and Phase-D ('roll'
ablation) results are not comparable.

The two files are intentionally NOT byte-identical: the ablation adds an
`if self.neg_type == 'roll' / else (random)` branch. This test confirms the
'roll' branch is operation-for-operation identical to the main loop by
comparing the parsed ASTs (no torch required — runs in CI).

    python repro/test_gnce_equivalence.py
"""
import ast
import os
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _extract_method(path, method):
    src = open(os.path.join(ROOT, path)).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method:
            fn = node
            # strip docstring
            if (fn.body and isinstance(fn.body[0], ast.Expr)
                    and isinstance(getattr(fn.body[0], "value", None), ast.Constant)):
                fn.body = fn.body[1:]
            return fn
    raise AssertionError(f"{method} not found in {path}")


def _specialize_roll(fn):
    """Replace `if self.neg_type == 'roll': <body> else: ...` with <body>."""
    new = []
    for node in fn.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and node.test.comparators
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "roll"):
            new.extend(node.body)
        else:
            new.append(node)
    fn.body = new
    return fn


def test_roll_path_identical():
    main = _extract_method("model/auxiliary_losses.py", "nce_loss")
    abl = _specialize_roll(
        _extract_method("model/auxiliary_losses_ablation.py", "nce_loss"))
    assert ast.dump(main) == ast.dump(abl), (
        "GNCELoss and GNCELossAblation 'roll' path diverged — Phase-B vs "
        "Phase-D comparison is INVALID until reconciled.")


if __name__ == "__main__":
    test_roll_path_identical()
    print("OK: GNCELoss 'roll' path is identical across the two files.")
    sys.exit(0)
