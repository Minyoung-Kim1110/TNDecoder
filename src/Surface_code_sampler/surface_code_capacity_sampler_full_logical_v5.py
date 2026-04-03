
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple
import numpy as np
import stim
try:
    from .stim_sampler import StimSurfaceSample, StimSurfaceBatchSample, _dense_syndrome_arrays_from_checks_batch, _rounded_detector_coords, _split_check_types_from_coords
except ImportError:
    from stim_sampler import StimSurfaceSample, StimSurfaceBatchSample, _dense_syndrome_arrays_from_checks_batch, _rounded_detector_coords, _split_check_types_from_coords

@dataclass(frozen=True)
class FullLogicalMetadata:
    distance:int; p:float; rounds:int; noisy_round:int; target_t:int; shots:int; num_data_qubits_in_noisy_round:int

@dataclass
class StimSurfaceBatchSampleFullLogical:
    batch: StimSurfaceBatchSample
    logical_bits: np.ndarray
    metadata: FullLogicalMetadata
    @property
    def shots(self)->int: return int(self.logical_bits.shape[0])
    def get_shot(self, shot:int)->Tuple[StimSurfaceSample,np.ndarray]:
        return self.batch.get_shot(shot), self.logical_bits[shot].astype(np.uint8, copy=False)
    def iter_shots(self)->Iterator[Tuple[StimSurfaceSample,np.ndarray]]:
        for k in range(self.shots): yield self.get_shot(k)

def _gf2_rref(A:np.ndarray):
    A=np.asarray(A,dtype=np.uint8).copy(); m,n=A.shape; piv=[]; r=0
    for c in range(n):
        pivot=None
        for rr in range(r,m):
            if A[rr,c]: pivot=rr; break
        if pivot is None: continue
        if pivot!=r: A[[r,pivot]]=A[[pivot,r]]
        for rr in range(m):
            if rr!=r and A[rr,c]: A[rr,:]^=A[r,:]
        piv.append(c); r+=1
        if r==m: break
    return A,piv
def _gf2_rank(A): return len(_gf2_rref(A)[1])
def _gf2_nullspace_basis(A):
    A=np.asarray(A,dtype=np.uint8); m,n=A.shape; rref,piv=_gf2_rref(A); pivs=set(piv); free=[c for c in range(n) if c not in pivs]; basis=[]
    for fc in free:
        v=np.zeros(n,dtype=np.uint8); v[fc]=1
        for row,pc in enumerate(piv):
            if rref[row,fc]: v[pc]=1
        basis.append(v)
    return basis
def _gf2_rowspace_basis(A):
    rref,_=_gf2_rref(np.asarray(A,dtype=np.uint8)); return [row.astype(np.uint8,copy=True) for row in rref if np.any(row)]
def _gf2_in_rowspace(v,row_basis):
    v=np.asarray(v,dtype=np.uint8).reshape(1,-1)
    if not row_basis: return not np.any(v)
    B=np.stack(row_basis,axis=0).astype(np.uint8)
    return _gf2_rank(B)==_gf2_rank(np.concatenate([B,v],axis=0))
def _choose_nontrivial_logical_from_css(null_basis,row_basis):
    for v in null_basis:
        if np.any(v) and not _gf2_in_rowspace(v,row_basis): return v.astype(np.uint8,copy=True)
    raise RuntimeError("Failed to find a nontrivial logical operator in the CSS quotient.")

def _validate_args(distance,p,rounds,noisy_round,target_t,shots):
    if distance<3 or distance%2==0: raise ValueError("distance must be an odd integer >= 3.")
    if not (0.0<=p<=1.0): raise ValueError("p must satisfy 0 <= p <= 1.")
    if rounds<3: raise ValueError("rounds must be >= 3.")
    if not (1<=noisy_round<=rounds): raise ValueError("noisy_round must satisfy 1 <= noisy_round <= rounds.")
    if noisy_round in (1,rounds): raise ValueError("Use an interior noisy_round, typically 2 when rounds=3.")
    if target_t!=noisy_round-1: raise ValueError("Use target_t = noisy_round - 1 for the single-round capacity slice.")
    if shots<=0: raise ValueError("shots must be positive.")

def _generated_marker_circuit(distance:int,rounds:int)->stim.Circuit:
    return stim.Circuit.generated("surface_code:unrotated_memory_x", distance=distance, rounds=rounds, after_clifford_depolarization=0.0, before_round_data_depolarization=0.125, before_measure_flip_probability=0.0, after_reset_flip_probability=0.0)

def _targets_of_noisy_round(circuit:stim.Circuit,noisy_round:int):
    occurrence=0
    for op in circuit.flattened():
        if op.name=="DEPOLARIZE1":
            occurrence+=1
            if occurrence==noisy_round: return op.targets_copy()
    raise RuntimeError(f"Could not find noisy_round={noisy_round} DEPOLARIZE1 occurrence in flattened circuit.")

def _rewrite_marker_circuit_with_explicit_paulis(circuit:stim.Circuit,*,noisy_round:int,pauli_pattern:np.ndarray)->stim.Circuit:
    out=stim.Circuit(); occurrence=0; pauli_pattern=np.asarray(pauli_pattern,dtype=np.uint8).reshape(-1)
    for op in circuit.flattened():
        if op.name!="DEPOLARIZE1":
            out.append(op); continue
        occurrence+=1
        if occurrence!=noisy_round: continue
        targets=op.targets_copy()
        if len(targets)!=int(pauli_pattern.size): raise RuntimeError(f"Mismatch between sampled pattern length and noisy-round target count: {len(pauli_pattern)} vs {len(targets)}.")
        x_targets=[t for t,code in zip(targets,pauli_pattern) if int(code)==1]
        y_targets=[t for t,code in zip(targets,pauli_pattern) if int(code)==2]
        z_targets=[t for t,code in zip(targets,pauli_pattern) if int(code)==3]
        if x_targets: out.append("X",x_targets)
        if y_targets: out.append("Y",y_targets)
        if z_targets: out.append("Z",z_targets)
    if occurrence==0 or occurrence<noisy_round: raise RuntimeError("Failed to locate the requested noisy round.")
    return out

def _sample_pauli_pattern(num_targets:int,p:float,rng:np.random.Generator)->np.ndarray:
    if p==0.0: return np.zeros(num_targets,dtype=np.uint8)
    probs=np.array([1.0-p,p/3.0,p/3.0,p/3.0],dtype=float)
    return rng.choice(4,size=num_targets,p=probs).astype(np.uint8)

def _dense_batch_from_circuit_sample(circuit:stim.Circuit,detector_bits:np.ndarray,observable_flips:np.ndarray,*,target_t:int)->StimSurfaceBatchSample:
    detector_coords: Dict[int,Tuple[int,int,int]]=_rounded_detector_coords(circuit)
    x_checks,z_checks=_split_check_types_from_coords(detector_coords=detector_coords,memory_basis="x",target_t=target_t)
    sX,sZ,active_X,active_Z=_dense_syndrome_arrays_from_checks_batch(detector_bits_batch=detector_bits,x_checks=x_checks,z_checks=z_checks)
    return StimSurfaceBatchSample(circuit=circuit,detector_bits=detector_bits,observable_flips=observable_flips,sX=sX,sZ=sZ,active_X=active_X,active_Z=active_Z,detector_coords=detector_coords)

def _single_qubit_error_pattern(num_targets:int,q:int,pauli_code:int)->np.ndarray:
    pat=np.zeros(num_targets,dtype=np.uint8); pat[q]=np.uint8(pauli_code); return pat

def _extract_css_matrices(marker:stim.Circuit,*,noisy_round:int,target_t:int,num_targets:int):
    zero_circuit=_rewrite_marker_circuit_with_explicit_paulis(marker,noisy_round=noisy_round,pauli_pattern=np.zeros(num_targets,dtype=np.uint8))
    zero_sampler=zero_circuit.compile_detector_sampler(); det0,obs0=zero_sampler.sample(shots=1,separate_observables=True)
    ref_batch=_dense_batch_from_circuit_sample(zero_circuit,np.asarray(det0,dtype=np.uint8).reshape(1,-1),np.asarray(obs0,dtype=np.uint8).reshape(1,-1),target_t=target_t)
    active_X=ref_batch.active_X[0].astype(np.uint8,copy=False); active_Z=ref_batch.active_Z[0].astype(np.uint8,copy=False)
    x_rows=np.where(active_X.reshape(-1)!=0)[0]; z_rows=np.where(active_Z.reshape(-1)!=0)[0]
    HX=np.zeros((len(x_rows),num_targets),dtype=np.uint8); HZ=np.zeros((len(z_rows),num_targets),dtype=np.uint8)
    for q in range(num_targets):
        circ_z=_rewrite_marker_circuit_with_explicit_paulis(marker,noisy_round=noisy_round,pauli_pattern=_single_qubit_error_pattern(num_targets,q,3))
        sampler_z=circ_z.compile_detector_sampler(); det_z,obs_z=sampler_z.sample(shots=1,separate_observables=True)
        bz=_dense_batch_from_circuit_sample(circ_z,np.asarray(det_z,dtype=np.uint8).reshape(1,-1),np.asarray(obs_z,dtype=np.uint8).reshape(1,-1),target_t=target_t)
        HX[:,q]=bz.sX[0].reshape(-1)[x_rows]
        circ_x=_rewrite_marker_circuit_with_explicit_paulis(marker,noisy_round=noisy_round,pauli_pattern=_single_qubit_error_pattern(num_targets,q,1))
        sampler_x=circ_x.compile_detector_sampler(); det_x,obs_x=sampler_x.sample(shots=1,separate_observables=True)
        bx=_dense_batch_from_circuit_sample(circ_x,np.asarray(det_x,dtype=np.uint8).reshape(1,-1),np.asarray(obs_x,dtype=np.uint8).reshape(1,-1),target_t=target_t)
        HZ[:,q]=bx.sZ[0].reshape(-1)[z_rows]
    return HX,HZ

def _logical_operators_from_css_matrices(HX:np.ndarray,HZ:np.ndarray):
    null_HX=_gf2_nullspace_basis(HX); null_HZ=_gf2_nullspace_basis(HZ); row_HX=_gf2_rowspace_basis(HX); row_HZ=_gf2_rowspace_basis(HZ)
    lx=_choose_nontrivial_logical_from_css(null_HX,row_HZ)
    lz=_choose_nontrivial_logical_from_css(null_HZ,row_HX)
    if int(np.dot(lx,lz)%2)!=1:
        for v in null_HZ:
            if np.any(v) and not _gf2_in_rowspace(v,row_HX):
                cand=(lz^v).astype(np.uint8)
                if int(np.dot(lx,cand)%2)==1 and not _gf2_in_rowspace(cand,row_HX):
                    lz=cand; break
    if int(np.dot(lx,lz)%2)!=1: raise RuntimeError("Failed to construct anticommuting logical X/Z operators from CSS matrices.")
    return lx.astype(np.uint8,copy=False), lz.astype(np.uint8,copy=False)

def _logical_bits_from_pauli_pattern(pauli_pattern:np.ndarray,*,lx:np.ndarray,lz:np.ndarray)->np.ndarray:
    pattern=np.asarray(pauli_pattern,dtype=np.uint8).reshape(-1)
    x_component=np.isin(pattern,np.array([1,2],dtype=np.uint8)).astype(np.uint8)
    z_component=np.isin(pattern,np.array([2,3],dtype=np.uint8)).astype(np.uint8)
    z_log=int(np.dot(z_component,lx)%2); x_log=int(np.dot(x_component,lz)%2)
    return np.array([z_log,x_log],dtype=np.uint8)

def sample_surface_code_capacity_batch_full_logical(distance:int,p:float,shots:int,*,rounds:int=3,noisy_round:int=2,target_t:int=1,seed:Optional[int]=None)->StimSurfaceBatchSampleFullLogical:
    _validate_args(distance,p,rounds,noisy_round,target_t,shots); rng=np.random.default_rng(seed)
    marker=_generated_marker_circuit(distance=distance,rounds=rounds); noisy_targets=_targets_of_noisy_round(marker,noisy_round=noisy_round); num_targets=len(noisy_targets)
    HX,HZ=_extract_css_matrices(marker,noisy_round=noisy_round,target_t=target_t,num_targets=num_targets)
    lx,lz=_logical_operators_from_css_matrices(HX,HZ)
    det_list=[]; obs_list=[]; logical_bits_list=[]; last_circuit=None
    for _ in range(shots):
        pauli_pattern=_sample_pauli_pattern(num_targets,p=p,rng=rng)
        logical_bits=_logical_bits_from_pauli_pattern(pauli_pattern,lx=lx,lz=lz)
        circuit=_rewrite_marker_circuit_with_explicit_paulis(marker,noisy_round=noisy_round,pauli_pattern=pauli_pattern)
        sampler=circuit.compile_detector_sampler(); dets,obs=sampler.sample(shots=1,separate_observables=True)
        det_list.append(np.asarray(dets,dtype=np.uint8).reshape(1,-1)); obs_list.append(np.asarray(obs,dtype=np.uint8).reshape(1,-1)); logical_bits_list.append(logical_bits); last_circuit=circuit
    detector_bits=np.concatenate(det_list,axis=0); observable_flips=np.concatenate(obs_list,axis=0); logical_bits=np.asarray(logical_bits_list,dtype=np.uint8)
    batch=_dense_batch_from_circuit_sample(last_circuit,detector_bits,observable_flips,target_t=target_t)
    return StimSurfaceBatchSampleFullLogical(batch=batch,logical_bits=logical_bits,metadata=FullLogicalMetadata(distance=distance,p=p,rounds=rounds,noisy_round=noisy_round,target_t=target_t,shots=shots,num_data_qubits_in_noisy_round=num_targets))

if __name__=="__main__":
    data=sample_surface_code_capacity_batch_full_logical(distance=5,p=0.02,shots=50,seed=1)
    print("true nontrivial fraction =", float(np.mean(np.any(data.logical_bits!=0,axis=1))))
