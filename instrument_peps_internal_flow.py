from __future__ import annotations

import argparse
import functools
import inspect
from types import FunctionType
from typing import Any, Dict, List, Tuple

import numpy as np

from src.Surface_code_sampler.surface_code_capacity_sampler_full_logical_v9 import (
    sample_surface_code_capacity_batch_full_logical_v9,
)
import src.ML_decoder_PEPS.PEPS_Pauli_decoder as peps_mod
from src.ML_decoder_PEPS.weights_PEPS import depolarizing_weights


KEYWORDS = [
    "pauli",
    "coset",
    "build",
    "reference",
    "recovery",
    "syndrome",
    "logical",
    "cut",
    "face",
    "tensor",
]


def arr_summary(x: np.ndarray, name: str = "") -> str:
    x = np.asarray(x)
    nz = np.argwhere(x != 0)
    nz_list = [tuple(map(int, p)) for p in nz[:12]]
    suffix = "" if len(nz) <= 12 else f" ... (+{len(nz)-12} more)"
    return (
        f"{name}shape={x.shape}, dtype={x.dtype}, "
        f"sum={float(np.sum(x))}, nnz={int(np.count_nonzero(x))}, nz={nz_list}{suffix}"
    )


def value_summary(v: Any, name: str = "") -> str:
    if isinstance(v, np.ndarray):
        return arr_summary(v, name=name)
    if isinstance(v, (list, tuple)):
        if len(v) <= 6 and all(not isinstance(t, (np.ndarray, list, tuple, dict)) for t in v):
            return f"{name}{type(v).__name__}{tuple(v)}"
        return f"{name}{type(v).__name__}(len={len(v)})"
    if isinstance(v, dict):
        keys = list(v.keys())
        return f"{name}dict(keys={keys})"
    return f"{name}{type(v).__name__}={v}"


def print_relevant_functions() -> List[str]:
    print("=" * 100)
    print("RELEVANT FUNCTIONS IN PEPS MODULE")
    print("=" * 100)
    found = []
    for name, obj in sorted(vars(peps_mod).items()):
        if isinstance(obj, FunctionType):
            lname = name.lower()
            if any(k in lname for k in KEYWORDS):
                found.append(name)
                try:
                    sig = inspect.signature(obj)
                except Exception:
                    sig = "(signature unavailable)"
                print(f"{name}{sig}")
    if not found:
        print("No relevant functions found by keyword scan.")
    return found


def wrap_function(name: str):
    obj = getattr(peps_mod, name, None)
    if not isinstance(obj, FunctionType):
        return False

    @functools.wraps(obj)
    def wrapped(*args, **kwargs):
        print("\n" + "-" * 100)
        print(f"ENTER {name}")
        print("-" * 100)

        try:
            sig = inspect.signature(obj)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            for k, v in bound.arguments.items():
                if isinstance(v, np.ndarray):
                    print("arg:", arr_summary(v, name=f"{k}: "))
                elif k in {"logical_x_cut_col", "logical_z_cut_row", "Nkeep", "Nsweep"}:
                    print("arg:", value_summary(v, name=f"{k}: "))
                elif k in {"W_h", "W_v"} and isinstance(v, np.ndarray):
                    print("arg:", arr_summary(v, name=f"{k}: "))
        except Exception as exc:
            print(f"(argument introspection failed for {name}: {exc})")

        out = obj(*args, **kwargs)

        print(f"EXIT {name}")
        if isinstance(out, np.ndarray):
            print("ret:", arr_summary(out, name="return: "))
        elif isinstance(out, dict):
            print("ret:", value_summary(out, name="return: "))
            # If this looks like cosets, print values
            if all(isinstance(k, tuple) and len(k) == 2 for k in out.keys()):
                try:
                    print("ret cosets:", {k: float(out[k]) for k in sorted(out)})
                except Exception:
                    pass
        elif isinstance(out, tuple):
            print("ret:", value_summary(out, name="return: "))
            for i, item in enumerate(out):
                if isinstance(item, np.ndarray):
                    print("ret item:", arr_summary(item, name=f"return[{i}]: "))
        else:
            print("ret:", value_summary(out, name="return: "))
        return out

    setattr(peps_mod, name, wrapped)
    return True


def install_wrappers() -> List[str]:
    candidates = [
        # common/public
        "pauli_coset_likelihoods_peps",
        "build_pauli_peps",
        "choose_default_logical_cuts",
        # likely internal helpers
        "_build_face_tensor",
        "_build_face",
        "_reference_from_syndrome",
        "_build_reference_from_syndrome",
        "_build_recovery_from_syndrome",
        "_syndrome_to_reference",
        "_syndrome_to_recovery",
        "_apply_logical_twists",
        "_apply_logical_twist",
    ]
    wrapped = []
    for name in candidates:
        if wrap_function(name):
            wrapped.append(name)

    # also opportunistically wrap any function whose name matches keywords and contains
    # "reference", "recovery", or "syndrome"
    for name, obj in sorted(vars(peps_mod).items()):
        if not isinstance(obj, FunctionType):
            continue
        lname = name.lower()
        if name in wrapped:
            continue
        if any(k in lname for k in ["reference", "recovery", "syndrome"]):
            if wrap_function(name):
                wrapped.append(name)

    print("\nWrapped functions:")
    for w in wrapped:
        print(" ", w)
    if not wrapped:
        print("  (none)")
    return wrapped


def find_representatives(logical_bits: np.ndarray, sX_all: np.ndarray, sZ_all: np.ndarray):
    reps: Dict[Tuple[int, int], int | None] = {}
    for target in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        best = None
        best_w = None
        for k in range(len(logical_bits)):
            bits = tuple(map(int, logical_bits[k]))
            if bits != target:
                continue
            w = int(np.sum(sX_all[k]) + np.sum(sZ_all[k]))
            if best is None or w < best_w:
                best = k
                best_w = w
        reps[target] = best
    return reps


def run_one_shot(k: int, data, W_h, W_v, force_center: bool, Nkeep: int, Nsweep: int):
    sX = np.asarray(data.batch.sX[k], dtype=np.uint8)
    sZ = np.asarray(data.batch.sZ[k], dtype=np.uint8)
    active_X = np.asarray(data.batch.active_X[k], dtype=np.uint8)
    active_Z = np.asarray(data.batch.active_Z[k], dtype=np.uint8)
    bits = tuple(map(int, data.logical_bits[k]))

    nrow, ncol = sX.shape
    if force_center:
        cut_col = max(1, min(ncol // 2, ncol - 1))
        cut_row = max(1, min(nrow // 2, nrow - 1))
    else:
        cut_col, cut_row = peps_mod.choose_default_logical_cuts(active_X, active_Z)

    print("\n" + "=" * 100)
    print(f"RUN SHOT {k}  true bits={bits}")
    print("=" * 100)
    print(arr_summary(sX, "sX: "))
    print(arr_summary(sZ, "sZ: "))
    print(arr_summary(active_X, "active_X: "))
    print(arr_summary(active_Z, "active_Z: "))
    print(f"cuts: col={cut_col}, row={cut_row}")

    cosets = peps_mod.pauli_coset_likelihoods_peps(
        sX=sX,
        sZ=sZ,
        active_X=active_X,
        active_Z=active_Z,
        W_h=W_h,
        W_v=W_v,
        logical_x_cut_col=cut_col,
        logical_z_cut_row=cut_row,
        Nkeep=Nkeep,
        Nsweep=Nsweep,
    )
    print("\nFINAL COSETS:", {k: float(v) for k, v in sorted(cosets.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--p", type=float, default=0.01)
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--noisy-round", type=int, default=2)
    ap.add_argument("--target-t", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--force-central-cuts", action="store_true")
    ap.add_argument("--peps-nkeep", type=int, default=128)
    ap.add_argument("--peps-nsweep", type=int, default=1)
    ap.add_argument("--shot-a", type=int, default=None)
    ap.add_argument("--shot-b", type=int, default=None)
    args = ap.parse_args()

    print_relevant_functions()
    install_wrappers()

    data = sample_surface_code_capacity_batch_full_logical_v9(
        distance=args.distance,
        p=args.p,
        shots=args.shots,
        rounds=args.rounds,
        noisy_round=args.noisy_round,
        target_t=args.target_t,
        seed=args.seed,
    )

    logical_bits = np.asarray(data.logical_bits, dtype=np.uint8)
    sX_all = np.asarray(data.batch.sX, dtype=np.uint8)
    sZ_all = np.asarray(data.batch.sZ, dtype=np.uint8)
    reps = find_representatives(logical_bits, sX_all, sZ_all)
    print("\nRepresentatives:", reps)

    nrow, ncol = sX_all[0].shape
    W_h, W_v = depolarizing_weights(nrow, ncol, p=args.p)

    # default contrasting pair: lowest-syndrome X and Z representatives
    shot_a = args.shot_a if args.shot_a is not None else reps[(0, 1)]
    shot_b = args.shot_b if args.shot_b is not None else reps[(1, 0)]

    if shot_a is not None:
        run_one_shot(
            k=shot_a,
            data=data,
            W_h=W_h,
            W_v=W_v,
            force_center=args.force_central_cuts,
            Nkeep=args.peps_nkeep,
            Nsweep=args.peps_nsweep,
        )

    if shot_b is not None and shot_b != shot_a:
        run_one_shot(
            k=shot_b,
            data=data,
            W_h=W_h,
            W_v=W_v,
            force_center=args.force_central_cuts,
            Nkeep=args.peps_nkeep,
            Nsweep=args.peps_nsweep,
        )


if __name__ == "__main__":
    main()