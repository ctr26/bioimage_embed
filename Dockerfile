FROM mambaorg/micromamba
COPY . .
RUN micromamba install -f environment.yml -n base --yes
