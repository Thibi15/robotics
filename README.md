# Robotics (B-KUL-H02A4A) Software Environment

This README describes how to install the relevant software used in the exercise sessions. The software primarily relies on Python, Conda, Pinocchio, and CasADi.

The idea is that if you can run the notebook without errors, you should be ready for the course. This software environment will be used for all exercise sessions unless otherwise stated. 

## Install Miniconda

- Download and install Miniconda (or Conda) from: [Miniconda Installation Guide](https://docs.anaconda.com/miniconda/install/).

## Create and Activate a Conda Environment

- In the same directory as this README, run the following command to create an environment:
  ```bash
  conda env create -f py13roboticscourse_env.yml
  ```
  For windows, this command should be executed in the "Anaconda prompt".
- Activate the environment:
  ```bash
  conda activate py13roboticscourse
  ```

## Run a Notebook

### Visual Studio Code
Open the folder in Visual Studio Code
``` bash
code .
```
Open up the .ipynb file, and select the py13roboticscourse kernel. 
You should now be able to run the code.

### Jupyter lab
It is also possible to use Jupyter lab.
For this, first activate the conda environment, and then launch Jupyter Lab within the directory where the exercise is located.
  ```bash
  jupyter-lab
  ```
The notebook will be accessible from your web browser at [http://localhost:8888](http://localhost:8888).

If you can run all the cells without issues, you are ready to proceed!

## Troubleshooting

If you encounter issues installing or running the software, feel free to post a message on the Toledo forum. Similarly, if you know the answer to a question asked on the forum, please contribute by suggesting a solution.


## Docker

For those familiar with Docker, a [Dockerfile](Dockerfile) is available.

### Steps:
1. Install Docker.
2. Build the Docker image by running the following command in the same directory as the Dockerfile:
   ```bash
   docker build -t robotics_course_img .
   ```
3. Launch the Docker container within the directory of the exercise session:
   ```bash
   docker run --rm -v .:/app --net=host -it robotics_course_img
   ```
4. Look for the line starting with `http://127.0.0.1:8888/lab?token=...` in the output and open it in your browser to launch the notebook.

