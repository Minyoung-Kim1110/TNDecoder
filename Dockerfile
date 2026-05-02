FROM continuumio/miniconda3:latest

RUN conda create -n TNDecoder python=3.11 -y && \
    conda run -n TNDecoder pip install --no-cache-dir \
        numpy==2.2.6 \
        stim==1.15.0 \
        PyMatching==2.3.1 \
        matplotlib==3.10.8 \
        scipy==1.15.3

# Make conda activate work in bash scripts
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate TNDecoder" >> /etc/bash.bashrc

WORKDIR /workspace
