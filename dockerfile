# Use the Miniconda base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the environment yaml
COPY py13roboticscourse_env.yml .
# Install packages to new conda environment
RUN conda env create -f py13roboticscourse_env.yml 

# Update PATH so that the new environment's executables are used
ENV PATH=/opt/conda/envs/py13roboticscourse/bin:$PATH

# Expose the port Jupyter Notebook will run on
EXPOSE 8888

# Start Jupyter Notebook on container launch
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
