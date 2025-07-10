# Base image with Miniconda
FROM continuumio/miniconda3

WORKDIR /src

# Setup basic conda environment
ENV CONDA_ENVIRONMENT py_env
COPY environment.yml .

# Create the environment
RUN conda env create --file environment.yml --name ${CONDA_ENVIRONMENT} || \
    (cat /root/.conda/pkgs/*.log && false)

# Activate the environment on container start
RUN echo "source activate ${CONDA_ENVIRONMENT}" > ~/.bashrc && \
    conda clean --all -y

# Ensure the environment's bin directory is in the PATH
ENV PATH=/opt/conda/envs/${CONDA_ENVIRONMENT}/bin:$PATH

# For Makefiles
RUN apt-get update && apt-get install -y make

# Install main package
RUN rm -rf /src/*

# Copy all contents
WORKDIR /home
COPY . /home

# Load torch
RUN pip install torch

# Default shell
CMD [ "bash" ]
