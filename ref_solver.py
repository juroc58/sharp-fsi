"""
ref_solver.py
=============
Locate the independent reference BEM/CPPC solver that sharpFSI cross-checks.

sharpFSI deliberately does **not** vendor the code it validates.  The reference
solver is a separate, independently developed project:

    sharp-vawt-lab  --  https://github.com/juroc58/sharp-vawt-lab

whose physics core is the module ``vawt_core.py`` (BEM/DMST rotor model, CPPC
pitch dynamics, polar database).  Three places need it:

* ``compare_with_bem.py`` -- every head-to-head run;
* ``verify_solver.py``    -- check F10 (kinematic cross-check), skipped if absent;
* ``aero_correction.StaticPolar.naca0012`` -- the optional tabulated NACA 0012
  polar, which is one of the datasets shipped in that repository.

Resolution order (first hit wins), none of which is machine- or
username-specific:

1. the ``SHARP_VAWT_LAB`` environment variable (``SHARP_PROJ`` is still accepted
   as a legacy alias), pointing at the cloned repository root;
2. conventional clone locations -- a sibling or child directory called
   ``sharp-vawt-lab`` (or ``sharp_proj``), next to *this* file, its parent, the
   current directory, or the user's home.

If nothing is found, the callers either skip the cross-check or exit with a
message telling the user to clone the repository and set the variable.  Nothing
here ever references a specific machine's directory layout.

Usage::

    from ref_solver import find, polar_dir, INSTALL_HINT

    root = find()                 # absolute path, or None
    csvs = polar_dir(root)        # .../airfoil_polars/naca0012, or None
"""

from __future__ import annotations

import os

#: Environment variables consulted, in order (the second is a legacy alias).
ENV_VARS = ('SHARP_VAWT_LAB', 'SHARP_PROJ')

#: File that identifies a directory as the reference solver root.
MARKER = 'vawt_core.py'

#: Directory names the reference repository may have been cloned into.
DIR_NAMES = ('sharp-vawt-lab', 'sharp_proj')

#: NACA 0012 table locations, newest layout first.
POLAR_SUBDIRS = (('airfoil_polars', 'naca0012'), ('polars', 'naca0012'))
POLAR_PREFIX = 'polars_naca0012_Re_'

#: Relative roots tried in addition to ``DIR_NAMES``.
REL_PREFIXES = ('.', '..', os.path.join('..', '..'))

INSTALL_HINT = (
    "the reference solver sharp-vawt-lab is required but was not found.\n"
    "    git clone https://github.com/juroc58/sharp-vawt-lab\n"
    "    export SHARP_VAWT_LAB=\"$PWD/sharp-vawt-lab\"\n"
    "(alternatively clone it as a sibling directory called sharp-vawt-lab)")

_HERE = os.path.dirname(os.path.abspath(__file__))


def candidates():
    """Yield the paths that are searched, in order."""
    seen = set()
    raw = []
    for var in ENV_VARS:
        val = os.environ.get(var)
        if val:
            raw.append(val)
    for pre in REL_PREFIXES:
        for name in DIR_NAMES:
            raw.append(os.path.join(pre, name))
    for name in DIR_NAMES:
        raw.append(os.path.join(_HERE, '..', name))
        raw.append(os.path.join('~', name))
    for c in raw:
        p = os.path.abspath(os.path.expanduser(c))
        if p not in seen:
            seen.add(p)
            yield p


def is_root(path):
    """True if *path* looks like a clone of the reference solver."""
    return bool(path) and os.path.isfile(os.path.join(path, MARKER))


def find(extra=None):
    """Return the reference solver root as an absolute path, or ``None``.

    *extra* may be a path (or an iterable of paths) to try before the standard
    candidates, e.g. a user-supplied `--ref` argument.
    """
    if extra:
        extras = [extra] if isinstance(extra, str) else list(extra)
        for c in extras:
            p = os.path.abspath(os.path.expanduser(c))
            if is_root(p):
                return p
    for p in candidates():
        if is_root(p):
            return p
    return None


def polar_dir(root=None):
    """Return the directory holding the reference NACA 0012 tables, or ``None``.

    Accepts both the current ``airfoil_polars/naca0012`` layout and the older
    ``polars/naca0012`` one.
    """
    root = root or find()
    if not root:
        return None
    for rel in POLAR_SUBDIRS:
        d = os.path.join(root, *rel)
        try:
            names = os.listdir(d)
        except OSError:
            continue
        if any(f.startswith(POLAR_PREFIX) for f in names):
            return d
    return None


if __name__ == '__main__':                                   # pragma: no cover
    root = find()
    print(f"reference solver : {root or 'NOT FOUND'}")
    if root:
        print(f"polar tables     : {polar_dir(root) or 'NOT FOUND'}")
    print("searched:")
    for p in candidates():
        print(f"  {'OK ' if is_root(p) else '   '} {p}")
