FROM continuumio/miniconda3:latest

RUN conda create -n TN python=3.11 -y && \
    conda run -n TN pip install --no-cache-dir \
        numpy==2.2.6 \
        stim==1.15.0 \
        PyMatching==2.3.1 \
        matplotlib==3.10.8 \
        scipy==1.15.3

# Make conda activate work in bash scripts
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate TN" >> /etc/bash.bashrc

WORKDIR /workspace
