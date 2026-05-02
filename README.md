### 1. Elevator pitch 
My project aims to developing a noise-adapted, tensor-network–based maximum-likelihood decoder for quantum error correction, using a hardware motivated quasi-static (1/f) charge-noise model mapped to an effective (Z)-Pauli channel. I will focus on reproducing TN based ML decoder, then combine Stim-generated syndrome data with tensor-network contraction to study repetition and surface code. Furthremore, if possible, I plan to investigate how device level noise strength determin logical error rates and logical memory lifetime. 

### 2. Collaborator 
None. 

### 3. Short bullets

- **Project form:** Computational study with a reusable TN (tensor network)–based ML (maximum likelihood) decoder implementing hardware-motivated $Z$-noise models (quasi-static $1/f$ charge noise).
- **Baseline / comparison:** Compare TN ML decoding against standard minimum-weight decoding (and exact ML in 1D repetition codes) under identical $p_Z$ noise derived from a $1/f$ noise model.
- **Full-stack lever:** Demonstrate how hardware-level noise strength (charge-noise amplitude) quantitatively determines logical memory lifetime and the breakdown point of error correction.

### 4. One motivating reference

- Error correction decoder with tensor network   
   1. [S. Bravyi *et al.*, *Efficient algorithms for maximum likelihood decoding in the surface code*, **Phys. Rev. A** (2014)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.90.032326)
   2. [J. Darmawan, *Optimal adaptation of surface-code decoders to local noise*, **Phys. Rev. A** (2024)](https://journals.aps.org/pra/abstract/10.1103/r2dc-qcrx)
   3. [A. Ferris *et al.*, *Tensor Networks and Quantum Error Correction*, **Arxiv** (2014)](https://arxiv.org/pdf/1312.4578)
   4. [C. Gidney, *Stim: a fast stabilizer circuit simulator*, **Quantum** (2021)](https://quantum-journal.org/papers/q-2021-07-06-497/)
   5. [O. Higgott, *PyMatching: A python package for decoding quantum codes with minimum-weight perfect matching*, **Arxiv** (2021)](https://quantum-journal.org/papers/q-2021-07-06-497/)

- Charge noise in semiconductor device
   1. [J. R. Petta *et al.*, *Semiconductor spin qubits*, **Rev. Mod. Phys. 95, 025003** (2023)](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.95.025003)

