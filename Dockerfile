# The build-stage image:
# FROM continuumio/miniconda3 AS build
FROM continuumio/anaconda3 AS build

# Install the package as normal:
# COPY environment.yml .
WORKDIR /gee_pipeline
COPY ./environment.yml .
RUN conda env create -n peep -f ./environment.yml

# Install Google Cloud SDK (to obtain `gcloud` cmd)
COPY ./install_gcloud.sh .
RUN ./install_gcloud.sh
ENV PATH="${PATH}:/gee_pipeline/google-cloud-sdk/bin"

# # SHELL ["/bin/bash", "-c"]
# # ENTRYPOINT
SHELL ["conda", "run", "-n", "peep", "/bin/bash", "-c"]

# SHELL ["/bin/bash", "-c"]
RUN python -c "print('hello from build!')"

COPY . .

RUN python -m pip install .

# SHELL ["/bin/bash", "-c"]
ENTRYPOINT tail -f /dev/null

# RUN python -m pip install /gee_pipeline


# # Install conda-pack:
# RUN conda install -c conda-forge conda-pack

# # Use conda-pack to create a standalone enviornment
# # in /venv:
# RUN conda-pack -n peep -o /tmp/env.tar && \
#   mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
#   rm /tmp/env.tar

# # We've put venv in same path it'll be in final image,
# # so now fix up paths:
# RUN /venv/bin/conda-unpack


# # The runtime-stage image; we can use Debian as the
# # base image since the Conda env also includes Python
# # for us.
# FROM debian:buster AS runtime

# # Copy /venv from the previous stage:
# COPY --from=build /venv /venv

# # When image is run, run the code with the environment
# # activated:
# SHELL ["/bin/bash", "-c"]
# ENTRYPOINT source /venv/bin/activate && \
#            python -c "import numpy; print('hello from peep!')"
