import numpy as np

from src.stim_sampler import sample_surface_code_depolarizing_batch


# =============================================================================
# Utility: robust detector-coordinate access
# =============================================================================
def get_detector_coords_from_batch(batch):
    """
    Return detector coordinates as a numpy array of shape (num_detectors, k),
    with rows ordered by detector index.

    Supports:
      - batch.detector_coords as ndarray
      - batch.detector_coords as dict {detector_index: coord_tuple}
      - dicts nested under metadata-like fields
    """
    candidate_names = [
        "detector_coords",
        "det_coords",
        "coords",
        "detector_coordinates",
    ]

    def dict_to_coord_array(d, expected_n=None):
        """
        Convert a dict like {det_index: (x, y, t), ...}
        into an array coords[det_index] = [x, y, t, ...].
        """
        if len(d) == 0:
            raise RuntimeError("detector_coords dict is empty")

        # sort by detector index
        keys = sorted(d.keys())
        first_val = d[keys[0]]

        # values may be tuples/lists/arrays
        first_arr = np.asarray(first_val, dtype=float).ravel()
        k = len(first_arr)

        if expected_n is None:
            n = max(keys) + 1
        else:
            n = expected_n

        out = np.zeros((n, k), dtype=float)

        for det_idx, coord in d.items():
            arr = np.asarray(coord, dtype=float).ravel()
            if len(arr) != k:
                raise RuntimeError(
                    f"Inconsistent detector coord length for detector {det_idx}: "
                    f"expected {k}, got {len(arr)}"
                )
            out[int(det_idx), :] = arr

        return out

    # ------------------------------------------------------------------
    # direct attributes on batch
    # ------------------------------------------------------------------
    for name in candidate_names:
        if hasattr(batch, name):
            obj = getattr(batch, name)

            if isinstance(obj, np.ndarray):
                arr = np.asarray(obj)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    print(f"Using batch.{name} for detector coordinates")
                    return arr

            if isinstance(obj, dict):
                print(f"Using batch.{name} dict for detector coordinates")
                expected_n = None
                if hasattr(batch, "detector_bits"):
                    expected_n = np.asarray(batch.detector_bits).shape[1]
                return dict_to_coord_array(obj, expected_n=expected_n)

    # ------------------------------------------------------------------
    # metadata-like dict fields
    # ------------------------------------------------------------------
    for dname in ["meta", "metadata", "info", "sampler_metadata"]:
        if hasattr(batch, dname):
            obj = getattr(batch, dname)
            if isinstance(obj, dict):
                print(f"Inspecting dict batch.{dname}.keys() = {list(obj.keys())}")
                for key in candidate_names:
                    if key in obj:
                        sub = obj[key]

                        if isinstance(sub, np.ndarray):
                            arr = np.asarray(sub)
                            if arr.ndim == 2 and arr.shape[1] >= 3:
                                print(f"Using batch.{dname}['{key}'] for detector coordinates")
                                return arr

                        if isinstance(sub, dict):
                            print(f"Using batch.{dname}['{key}'] dict for detector coordinates")
                            expected_n = None
                            if hasattr(batch, "detector_bits"):
                                expected_n = np.asarray(batch.detector_bits).shape[1]
                            return dict_to_coord_array(sub, expected_n=expected_n)

    print("\nCould not find detector coordinates automatically.")
    print("Available batch.__dict__.keys():")
    for k in batch.__dict__.keys():
        v = getattr(batch, k)
        print("  ", k, type(v), getattr(v, "shape", None))

    raise RuntimeError(
        "Could not find detector coordinates on batch even after dict conversion."
    )
# def get_detector_coords_from_batch(batch):
#     """
#     Try several likely attribute names for detector coordinates.
#     Expected final shape: (num_detectors, k) with k>=3 and columns [x, y, t, ...]
#     """
#     candidate_names = [
#         "detector_coords",
#         "det_coords",
#         "coords",
#         "detector_coordinates",
#     ]

#     for name in candidate_names:
#         if hasattr(batch, name):
#             arr = np.asarray(getattr(batch, name))
#             if arr.ndim == 2 and arr.shape[1] >= 3:
#                 print(f"Using batch.{name} for detector coordinates")
#                 return arr

#     for dname in ["meta", "metadata", "info", "sampler_metadata"]:
#         if hasattr(batch, dname):
#             obj = getattr(batch, dname)
#             if isinstance(obj, dict):
#                 print(f"Inspecting dict batch.{dname}.keys() = {list(obj.keys())}")
#                 for key in candidate_names:
#                     if key in obj:
#                         arr = np.asarray(obj[key])
#                         if arr.ndim == 2 and arr.shape[1] >= 3:
#                             print(f"Using batch.{dname}['{key}'] for detector coordinates")
#                             return arr

#     print("\nCould not find detector coordinates automatically.")
#     print("Available batch.__dict__.keys():")
#     for k in batch.__dict__.keys():
#         v = getattr(batch, k)
#         print("  ", k, type(v), getattr(v, "shape", None))

#     raise RuntimeError(
#         "Could not find detector coordinates on batch. "
#         "Paste the printed keys and I will patch the script to the correct field."
#     )
# def get_detector_coords_from_batch(batch):
#     """
#     Try several likely attribute names for detector coordinates.
#     Expected final shape: (num_detectors, k) with k>=3 and columns [x, y, t, ...]
#     """
#     candidate_names = [
#         "detector_coords",
#         "det_coords",
#         "coords",
#         "detector_coordinates",
#     ]

#     for name in candidate_names:
#         if hasattr(batch, name):
#             arr = np.asarray(getattr(batch, name))
#             if arr.ndim == 2 and arr.shape[1] >= 3:
#                 return arr

#     # try dict-like metadata fields
#     candidate_dicts = ["meta", "metadata", "info", "__dict__"]
#     for dname in candidate_dicts:
#         if hasattr(batch, dname):
#             obj = getattr(batch, dname)
#             if isinstance(obj, dict):
#                 for key in candidate_names:
#                     if key in obj:
#                         arr = np.asarray(obj[key])
#                         if arr.ndim == 2 and arr.shape[1] >= 3:
#                             return arr

#     raise RuntimeError(
#         "Could not find detector coordinates on batch. "
#         "Please inspect `print(batch.__dict__.keys())` and add the correct field name."
#     )


# =============================================================================
# Step 1: split detector sites into two check families using only coordinates
# =============================================================================

def split_check_types_from_coords_manual(detector_coords, target_t=1):
    """
    Manual version of the detector-family split.

    Strategy:
      - first_round_xy = all (x,y) positions appearing at t=0
      - target detectors = all detectors at t=target_t
      - a_type = target detectors whose (x,y) already appeared at t=0
      - b_type = target detectors whose (x,y) are new at target_t

    This mirrors the sampler heuristic, but we compute it explicitly from raw coords.
    """
    det = np.asarray(detector_coords)
    x = det[:, 0].astype(int)
    y = det[:, 1].astype(int)
    t = det[:, 2].astype(int)

    first_round_xy = {(int(xx), int(yy)) for xx, yy, tt in zip(x, y, t) if tt == 0}

    target_idx = np.where(t == target_t)[0]
    target_xy = [(int(x[i]), int(y[i])) for i in target_idx]

    a_idx = []
    b_idx = []
    for idx, xy in zip(target_idx, target_xy):
        if xy in first_round_xy:
            a_idx.append(int(idx))
        else:
            b_idx.append(int(idx))

    return {
        "target_idx": np.array(target_idx, dtype=int),
        "a_idx": np.array(a_idx, dtype=int),
        "b_idx": np.array(b_idx, dtype=int),
        "first_round_xy": first_round_xy,
    }


# =============================================================================
# Step 2: build sparse checks from detector bits at one shot
# =============================================================================

def sparse_checks_from_detector_bits(detector_bits_1shot, detector_coords, det_indices):
    """
    Return sparse list of fired checks as [(x,y,bit), ...] for selected detector indices.
    """
    bits = np.asarray(detector_bits_1shot).astype(np.uint8)
    coords = np.asarray(detector_coords)

    out = []
    for i in det_indices:
        xx = int(coords[i, 0])
        yy = int(coords[i, 1])
        bb = int(bits[i])
        out.append((xx, yy, bb))
    return out


# =============================================================================
# Step 3: dense embedding from sparse checks
# =============================================================================

def dense_arrays_from_sparse_checks(x_checks, z_checks):
    """
    Build dense rectangular arrays from sparse coordinate lists.

    x_checks, z_checks are lists of (x, y, bit).

    Returns:
      sX, sZ, active_X, active_Z, x_map, y_map
    """
    used_xy = set()
    for xx, yy, _ in x_checks:
        used_xy.add((xx, yy))
    for xx, yy, _ in z_checks:
        used_xy.add((xx, yy))

    if not used_xy:
        raise RuntimeError("No checks supplied; cannot build dense arrays.")

    xs = sorted({xx for xx, yy in used_xy})
    ys = sorted({yy for xx, yy in used_xy})

    x_to_col = {xx: j for j, xx in enumerate(xs)}
    y_to_row = {yy: i for i, yy in enumerate(ys)}

    nrow = len(ys)
    ncol = len(xs)

    sX = np.zeros((nrow, ncol), dtype=np.uint8)
    sZ = np.zeros((nrow, ncol), dtype=np.uint8)
    active_X = np.zeros((nrow, ncol), dtype=np.uint8)
    active_Z = np.zeros((nrow, ncol), dtype=np.uint8)

    for xx, yy, bb in x_checks:
        r = y_to_row[yy]
        c = x_to_col[xx]
        sX[r, c] = bb
        active_X[r, c] = 1

    for xx, yy, bb in z_checks:
        r = y_to_row[yy]
        c = x_to_col[xx]
        sZ[r, c] = bb
        active_Z[r, c] = 1

    return sX, sZ, active_X, active_Z, x_to_col, y_to_row


# =============================================================================
# Step 4: comparison helpers
# =============================================================================

def compare_arrays(name, arr_manual, arr_sampler):
    arr_manual = np.asarray(arr_manual, dtype=np.uint8)
    arr_sampler = np.asarray(arr_sampler, dtype=np.uint8)

    if arr_manual.shape != arr_sampler.shape:
        print(f"{name}: SHAPE MISMATCH manual={arr_manual.shape}, sampler={arr_sampler.shape}")
        return

    diff = arr_manual ^ arr_sampler
    nbad = int(np.sum(diff))
    print(f"{name}: mismatch count = {nbad}")

    if nbad > 0:
        bad_pos = np.argwhere(diff)
        print(f"  first mismatches ({min(12, len(bad_pos))} shown):")
        for r, c in bad_pos[:12]:
            print(
                f"    (r={r}, c={c}) manual={int(arr_manual[r,c])} sampler={int(arr_sampler[r,c])}"
            )


def print_sparse_list(title, checks, max_items=20):
    print(title)
    for item in checks[:max_items]:
        print("  ", item)
    if len(checks) > max_items:
        print(f"  ... ({len(checks) - max_items} more)")


def print_dense_summary(label, sX, sZ, active_X, active_Z):
    print(f"{label}")
    print(f"  shape            : {sX.shape}")
    print(f"  sum sX           : {int(np.sum(sX))}")
    print(f"  sum sZ           : {int(np.sum(sZ))}")
    print(f"  active_X count   : {int(np.sum(active_X))}")
    print(f"  active_Z count   : {int(np.sum(active_Z))}")


# =============================================================================
# Main audit
# =============================================================================

def audit_peps_formatted_syndrome_interface(
    distance=5,
    p=0.01,
    shots=5,
    memory_basis="x",
    rounds=3,
    target_t=1,
    inspect_shot=0,
):
    print("=" * 100)
    print("PEPS-FORMATTED SYNDROME INTERFACE AUDIT")
    print("=" * 100)
    print(
        f"distance={distance}, p={p}, shots={shots}, memory_basis={memory_basis}, "
        f"rounds={rounds}, target_t={target_t}, inspect_shot={inspect_shot}"
    )

    batch = sample_surface_code_depolarizing_batch(
        distance=distance,
        p=p,
        shots=shots,
        memory_basis=memory_basis,
        rounds=rounds,
        target_t=target_t,
    )
    print(type(batch))
    print("batch.__dict__.keys() =")
    for k in batch.__dict__.keys():
        print("  ", k, type(getattr(batch, k)))
    for name in ["meta", "metadata", "circuit", "dem", "sampler_metadata"]:
        if hasattr(batch, name):
            obj = getattr(batch, name)
            print(f"\n{name} exists: {type(obj)}")
            if isinstance(obj, dict):
                print(f"{name}.keys() = {list(obj.keys())}")

    detector_coords = get_detector_coords_from_batch(batch)
    print("detector_coords array shape:", detector_coords.shape)
    print("first 10 detector coords:")
    for i in range(min(10, len(detector_coords))):
        print(i, detector_coords[i])
    detector_bits = np.asarray(batch.detector_bits, dtype=np.uint8)

    print("\nBatch-level info")
    print(f"  detector_bits shape    : {detector_bits.shape}")
    print(f"  detector_coords shape  : {detector_coords.shape}")
    print(f"  sampler sX shape       : {batch.sX.shape}")
    print(f"  sampler sZ shape       : {batch.sZ.shape}")

    split = split_check_types_from_coords_manual(detector_coords, target_t=target_t)

    print("\nCoordinate split summary")
    print(f"  # target detectors at t={target_t} : {len(split['target_idx'])}")
    print(f"  # a_type detectors                 : {len(split['a_idx'])}")
    print(f"  # b_type detectors                 : {len(split['b_idx'])}")

    # inspect one shot
    bits_1shot = detector_bits[inspect_shot]

    a_sparse = sparse_checks_from_detector_bits(bits_1shot, detector_coords, split["a_idx"])
    b_sparse = sparse_checks_from_detector_bits(bits_1shot, detector_coords, split["b_idx"])

    print("\nSparse checks for inspected shot")
    print_sparse_list("a_type sparse checks (x, y, bit):", a_sparse)
    print_sparse_list("b_type sparse checks (x, y, bit):", b_sparse)

    # -------------------------------------------------------------------------
    # Hypothesis A:
    #   a_type -> X checks
    #   b_type -> Z checks
    # -------------------------------------------------------------------------
    sX_A, sZ_A, active_X_A, active_Z_A, _, _ = dense_arrays_from_sparse_checks(a_sparse, b_sparse)

    # -------------------------------------------------------------------------
    # Hypothesis B:
    #   a_type -> Z checks
    #   b_type -> X checks
    # -------------------------------------------------------------------------
    sX_B, sZ_B, active_X_B, active_Z_B, _, _ = dense_arrays_from_sparse_checks(b_sparse, a_sparse)

    sX_sampler = np.asarray(batch.sX[inspect_shot], dtype=np.uint8)
    sZ_sampler = np.asarray(batch.sZ[inspect_shot], dtype=np.uint8)
    active_X_sampler = np.asarray(batch.active_X[inspect_shot], dtype=np.uint8)
    active_Z_sampler = np.asarray(batch.active_Z[inspect_shot], dtype=np.uint8)

    print("\nDense summaries")
    print_dense_summary("Hypothesis A (a->X, b->Z)", sX_A, sZ_A, active_X_A, active_Z_A)
    print_dense_summary("Hypothesis B (a->Z, b->X)", sX_B, sZ_B, active_X_B, active_Z_B)
    print_dense_summary("Sampler output", sX_sampler, sZ_sampler, active_X_sampler, active_Z_sampler)

    print("\nCompare hypothesis A to sampler")
    compare_arrays("sX_A vs batch.sX", sX_A, sX_sampler)
    compare_arrays("sZ_A vs batch.sZ", sZ_A, sZ_sampler)
    compare_arrays("active_X_A vs batch.active_X", active_X_A, active_X_sampler)
    compare_arrays("active_Z_A vs batch.active_Z", active_Z_A, active_Z_sampler)

    print("\nCompare hypothesis B to sampler")
    compare_arrays("sX_B vs batch.sX", sX_B, sX_sampler)
    compare_arrays("sZ_B vs batch.sZ", sZ_B, sZ_sampler)
    compare_arrays("active_X_B vs batch.active_X", active_X_B, active_X_sampler)
    compare_arrays("active_Z_B vs batch.active_Z", active_Z_B, active_Z_sampler)

    # Pretty print actual arrays for direct eyeballing
    np.set_printoptions(linewidth=200)

    print("\nSampler arrays (inspected shot)")
    print("batch.sX =")
    print(sX_sampler)
    print("batch.sZ =")
    print(sZ_sampler)
    print("batch.active_X =")
    print(active_X_sampler)
    print("batch.active_Z =")
    print(active_Z_sampler)

    print("\nHypothesis A arrays")
    print("sX_A =")
    print(sX_A)
    print("sZ_A =")
    print(sZ_A)
    print("active_X_A =")
    print(active_X_A)
    print("active_Z_A =")
    print(active_Z_A)

    print("\nHypothesis B arrays")
    print("sX_B =")
    print(sX_B)
    print("sZ_B =")
    print(sZ_B)
    print("active_X_B =")
    print(active_X_B)
    print("active_Z_B =")
    print(active_Z_B)

    return {
        "batch": batch,
        "detector_coords": detector_coords,
        "split": split,
        "A": (sX_A, sZ_A, active_X_A, active_Z_A),
        "B": (sX_B, sZ_B, active_X_B, active_Z_B),
        "sampler": (sX_sampler, sZ_sampler, active_X_sampler, active_Z_sampler),
    }


if __name__ == "__main__":
    print("\n" + "#" * 100)
    print("# MEMORY X")
    print("#" * 100)
    out_x = audit_peps_formatted_syndrome_interface(
        distance=5,
        p=0.01,
        shots=5,
        memory_basis="x",
        rounds=3,
        target_t=1,
        inspect_shot=0,
    )

    print("\n" + "#" * 100)
    print("# MEMORY Z")
    print("#" * 100)
    out_z = audit_peps_formatted_syndrome_interface(
        distance=5,
        p=0.01,
        shots=5,
        memory_basis="z",
        rounds=3,
        target_t=1,
        inspect_shot=0,
    )