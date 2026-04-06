import numpy as np
import matplotlib.pyplot as plt

from src.Surface_code_sampler.surface_code_capacity_sampler_full_logical_v9 import (
    sample_surface_code_capacity_batch_full_logical,
)
from src.MWPM_decoder_pymatching.mwpm_decoder_2d import (
    decode_2d_surface_batch_with_mwpm,
)


def wilson_half_width(num_fail: int, shots: int, z: float = 1.96) -> float:
    if shots == 0:
        return 0.0
    phat = num_fail / shots
    denom = 1.0 + z**2 / shots
    radius = (
        z
        * np.sqrt((phat * (1.0 - phat) / shots) + (z**2 / (4 * shots**2)))
        / denom
    )
    return float(radius)


def get_basis_batch(full_batch, memory_basis: str):
    """
    Convert v9 full-logical batch into the basis-specific StimSurfaceBatchSample
    expected by the 2D MWPM decoder.
    """
    if memory_basis == "x":
        return full_batch.batch_x
    if memory_basis == "z":
        return full_batch.batch_z
    raise ValueError("memory_basis must be 'x' or 'z'.")


def run_one_point(
    distance: int,
    p: float,
    shots: int,
    memory_basis: str,
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: int = 1,
):
    """
    One data point:
      Stim full-logical sampler -> pick basis-specific batch -> 2D MWPM decode
    """
    full_batch = sample_surface_code_capacity_batch_full_logical(
        distance=distance,
        rounds=rounds,
        noisy_round=noisy_round,
        target_t=target_t,
        shots=shots,
        p=p,
    )

    basis_batch = get_basis_batch(full_batch, memory_basis)

    out = decode_2d_surface_batch_with_mwpm(
        basis_batch,
        p=max(float(p), 1e-15),   # avoid q=0 issue in MWPM weights
        memory_basis=memory_basis,
    )

    num_fail = int(np.sum(out.logical_failures))
    pL = num_fail / shots
    err = wilson_half_width(num_fail, shots)

    return {
        "p": float(p),
        "pL": float(pL),
        "err": float(err),
        "num_fail": int(num_fail),
        "full_batch": full_batch,
        "basis_batch": basis_batch,
        "decoder_output": out,
    }


def run_sweep(
    distances=(3, 5, 7),
    p_values=None,
    shots=20000,
    memory_basis="x",
    rounds=3,
    noisy_round=2,
    target_t=1,
):
    if p_values is None:
        p_values = np.array([5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 3e-2], dtype=float)

    results = {}

    for d in distances:
        results[d] = {
            "p": [],
            "pL": [],
            "err": [],
            "num_fail": [],
        }

        print(f"\n=== distance={d}, basis={memory_basis} ===")
        for p in p_values:
            r = run_one_point(
                distance=d,
                p=float(p),
                shots=shots,
                memory_basis=memory_basis,
                rounds=rounds,
                noisy_round=noisy_round,
                target_t=target_t,
            )

            results[d]["p"].append(r["p"])
            results[d]["pL"].append(r["pL"])
            results[d]["err"].append(r["err"])
            results[d]["num_fail"].append(r["num_fail"])

            print(
                f"p={p:8.5g} | pL={r['pL']:10.6g} | "
                f"failures={r['num_fail']:6d}/{shots}"
            )

        results[d]["p"] = np.array(results[d]["p"], dtype=float)
        results[d]["pL"] = np.array(results[d]["pL"], dtype=float)
        results[d]["err"] = np.array(results[d]["err"], dtype=float)

    return results


def plot_results(results, memory_basis="x"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    all_p = np.concatenate([results[d]["p"] for d in results])
    pmin = float(np.min(all_p))
    pmax = float(np.max(all_p))
    pref = np.geomspace(pmin, pmax, 200)

    ax.loglog(pref, pref, "--", linewidth=1.5, label=r"$p_L=p$")

    for d in sorted(results):
        ax.errorbar(
            results[d]["p"],
            results[d]["pL"],
            yerr=results[d]["err"],
            marker="o",
            linestyle="-",
            capsize=3,
            label=f"d={d}",
        )

    ax.set_xlabel("Physical error rate $p$")
    ax.set_ylabel("Logical error rate $p_L$")
    ax.set_title(f"2D MWPM sanity check ({memory_basis}-memory)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def quick_sanity_check():
    """
    Zero-noise sanity check for both X-memory and Z-memory.
    """
    print("Running zero-noise sanity checks...")

    full_batch = sample_surface_code_capacity_batch_full_logical(
        distance=5,
        rounds=3,
        noisy_round=2,
        target_t=1,
        shots=200,
        p=0.0,
    )

    for basis in ("x", "z"):
        basis_batch = get_basis_batch(full_batch, basis)

        out = decode_2d_surface_batch_with_mwpm(
            basis_batch,
            p=1e-15,
            memory_basis=basis,
        )

        num_fail = int(np.sum(out.logical_failures))
        print(f"basis={basis}: zero-noise failures = {num_fail}/200")


if __name__ == "__main__":
    quick_sanity_check()

    p_values = np.array([
        5e-4,
        1e-3,
        2e-3,
        5e-3,
        1e-2,
        2e-2,
        3e-2,
    ], dtype=float)

    distances = (3, 5, 7)
    shots = 20

    results_x = run_sweep(
        distances=distances,
        p_values=p_values,
        shots=shots,
        memory_basis="x",
        rounds=3,
        noisy_round=2,
        target_t=1,
    )
    plot_results(results_x, memory_basis="x")

    results_z = run_sweep(
        distances=distances,
        p_values=p_values,
        shots=shots,
        memory_basis="z",
        rounds=3,
        noisy_round=2,
        target_t=1,
    )
    plot_results(results_z, memory_basis="z")