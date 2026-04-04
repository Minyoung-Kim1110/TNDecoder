
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import numpy as np

from src.Surface_code_sampler.surface_code_capacity_sampler_full_logical_v6 import (
    StimSurfaceBatchSampleFullLogical,
    sample_surface_code_capacity_batch_full_logical,
)
from src.ML_decoder_PEPS.PEPS_Pauli_decoder import (
    peps_coset_likelihoods_for_batch,
    most_likely_coset,
    _logical_bits_from_ml_coset as peps_bits_from_ml_coset
)
from src.MWPM_decoder_pymatching.mwpm_decoder_2d import (
    build_2d_css_matching,
    decode_2d_css_slice,
)
@property
def batch(self) -> StimSurfaceBatchSample:
    return StimSurfaceBatchSample(
        circuit=self.batch_x.circuit,
        detector_bits=self.batch_x.detector_bits,
        observable_flips=self.logical_bits,
        sX=self.batch_x.sX,
        sZ=self.batch_z.sZ,
        active_X=self.batch_x.active_X,
        active_Z=self.batch_z.active_Z,
        detector_coords=self.batch_x.detector_coords,
    )
    

@dataclass(frozen=True)
class DecoderFullLogicalPoint:
    p: float; shots:int; peps_pI:float; peps_pX:float; peps_pZ:float; peps_pY:float; peps_fail:float; peps_favg:float; mwpm_pI:float; mwpm_pX:float; mwpm_pZ:float; mwpm_pY:float; mwpm_fail:float; mwpm_favg:float; peps_vs_mwpm_agreement_rate:float; true_nontrivial_fraction:float
@dataclass
class DecoderFullLogicalTable:
    points: List[DecoderFullLogicalPoint]
    def pretty_print(self)->None:
        header=f"{'p':>8} | {'PEPS pI':>8} | {'MWPM pI':>8} | {'PEPS fail':>9} | {'MWPM fail':>9} | {'PEPS Favg':>10} | {'MWPM Favg':>10} | {'Agree':>7}"
        print(header); print("-"*len(header))
        for pt in self.points:
            print(f"{pt.p:8.4g} | {pt.peps_pI:8.4f} | {pt.mwpm_pI:8.4f} | {pt.peps_fail:9.4f} | {pt.mwpm_fail:9.4f} | {pt.peps_favg:10.6f} | {pt.mwpm_favg:10.6f} | {pt.peps_vs_mwpm_agreement_rate:7.4f}")

def _bits_to_label(bits):
    z,x=int(bits[0]),int(bits[1]); return "I" if (z,x)==(0,0) else "Z" if (z,x)==(1,0) else "X" if (z,x)==(0,1) else "Y"
def _residual_distribution(pred_bits,true_bits):
    residual=np.bitwise_xor(np.asarray(pred_bits,dtype=np.uint8),np.asarray(true_bits,dtype=np.uint8))
    labels=np.array([_bits_to_label(row) for row in residual],dtype=object)
    return {lab: float(np.mean(labels==lab)) if len(labels) else 0.0 for lab in ("I","X","Z","Y")}
def _favg_from_pI(pI:float)->float: return (1.0+2.0*float(pI))/3.0
def _predict_peps_logical_bits(data:StimSurfaceBatchSampleFullLogical,*,p:float,peps_nkeep:int,peps_nsweep:int)->np.ndarray:
    coset_likelihoods=peps_coset_likelihoods_for_batch(batch=data.batch,p=p,Nkeep=peps_nkeep,Nsweep=peps_nsweep)
    ml_cosets=[most_likely_coset(c) for c in coset_likelihoods]
    return np.asarray([peps_bits_from_ml_coset(c) for c in ml_cosets],dtype=np.uint8)

def _predict_mwpm_logical_bits(data:StimSurfaceBatchSampleFullLogical,*,p:float)->np.ndarray:
    batch=data.batch; shots=batch.shots
    matching_x,node_of_x=build_2d_css_matching(batch.active_X[0],p=p,boundary_axis="horizontal")
    matching_z,node_of_z=build_2d_css_matching(batch.active_Z[0],p=p,boundary_axis="vertical")
    pred_z=np.stack([decode_2d_css_slice(batch.sX[k],batch.active_X[k],p=p,boundary_axis="horizontal",matching=matching_x,node_of=node_of_x) for k in range(shots)],axis=0).astype(np.uint8)
    pred_x=np.stack([decode_2d_css_slice(batch.sZ[k],batch.active_Z[k],p=p,boundary_axis="vertical",matching=matching_z,node_of=node_of_z) for k in range(shots)],axis=0).astype(np.uint8)
    return np.concatenate([pred_z,pred_x],axis=1).astype(np.uint8)

def compare_peps_mwpm_surface_code_full_logical(*,distance:int,p_values:Iterable[float],shots:int,rounds:int=3,noisy_round:int=2,target_t:int=1,peps_nkeep:int=128,peps_nsweep:int=1,seed:Optional[int]=None)->DecoderFullLogicalTable:
    points=[]; seed_seq=np.random.SeedSequence(seed)
    for p in list(p_values):
        child_seed=int(seed_seq.spawn(1)[0].generate_state(1)[0]) if seed is not None else None
        data=sample_surface_code_capacity_batch_full_logical(distance=distance,p=p,shots=shots,rounds=rounds,noisy_round=noisy_round,target_t=target_t,seed=child_seed)
        true_bits=data.logical_bits.astype(np.uint8,copy=False)
        peps_pred=_predict_peps_logical_bits(data,p=p,peps_nkeep=peps_nkeep,peps_nsweep=peps_nsweep)
        mwpm_pred=_predict_mwpm_logical_bits(data,p=p)
        peps_dist=_residual_distribution(peps_pred,true_bits); mwpm_dist=_residual_distribution(mwpm_pred,true_bits)
        points.append(DecoderFullLogicalPoint(p=float(p),shots=int(shots),peps_pI=peps_dist["I"],peps_pX=peps_dist["X"],peps_pZ=peps_dist["Z"],peps_pY=peps_dist["Y"],peps_fail=1.0-peps_dist["I"],peps_favg=_favg_from_pI(peps_dist["I"]),mwpm_pI=mwpm_dist["I"],mwpm_pX=mwpm_dist["X"],mwpm_pZ=mwpm_dist["Z"],mwpm_pY=mwpm_dist["Y"],mwpm_fail=1.0-mwpm_dist["I"],mwpm_favg=_favg_from_pI(mwpm_dist["I"]),peps_vs_mwpm_agreement_rate=float(np.mean(np.all(peps_pred==mwpm_pred,axis=1))),true_nontrivial_fraction=float(np.mean(np.any(true_bits!=0,axis=1)))))
    return DecoderFullLogicalTable(points)

if __name__=="__main__":
    table=compare_peps_mwpm_surface_code_full_logical(distance=5,p_values=[0.002,0.005,0.01,0.02],shots=200,seed=1)
    table.pretty_print()
