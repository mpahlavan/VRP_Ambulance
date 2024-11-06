Here’s a refined and enhanced version of your README title and content:

---

# Optimized VRP Solution for Ambulance Dispatch in Critical Scenarios

## Overview

This project focuses on solving the **Vehicle Routing Problem (VRP)** with a machine learning approach, specifically tailored for **ambulance dispatch in critical conditions**. By using Docker, this solution ensures a streamlined and consistent environment for development, training, and deployment, removing dependency concerns.

The project includes a complete suite of scripts for training, evaluating, and visualizing the performance of a machine learning model that optimizes ambulance routes under emergency constraints.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup & How to Run](#setup--how-to-run)
- [Training](#training)
- [Project Workflow](#project-workflow)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before running this project, ensure the following are installed on your system:
- **Docker Desktop**: For containerization and environment management.
- **Visual Studio Code (VSCode)**: As the primary IDE.
- **Dev Containers Extension**: To interact seamlessly with the Docker container in VSCode.

You can install the VSCode Dev Containers extension [here](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

## Setup & How to Run

Follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/mpahlavan/VRP_Ambulance.git
   cd VRP_Ambulance
   ```

2. Open the project in **Visual Studio Code**:
   ```bash
   code .
   ```

3. Build the Docker image from within VSCode’s terminal:
   ```bash
   docker build -t vrp_ambulance_image -f docker/Dockerfile .
   ```

4. Verify the image was created by listing the Docker images:
   ```bash
   docker image ls
   ```

5. Run the Docker container interactively:
   ```bash
   docker run -it --name vrp_ambulance_container vrp_ambulance_image
   ```

6. Attach to the running container from VSCode:
   - Press **Ctrl+Shift+P**.
   - Select **Dev Container: Attach to Running Container...**.
   - Choose **/vrp_ambulance_container**.
   - open folder **./py_ws/marpdan/**.
   
You’re now inside the Docker container, ready to execute the project code directly in this isolated environment.

## Project Workflow

To run the full sequence of processes, follow the steps below:

### Configuration
```bash
echo "Generating configurations..."
python cfgs/gen_cfgs.py
```

### Data Generation
```bash
echo "Generating validation data..."
python script/gen_val_data.py
```

### Training
```bash
echo "Training the model..."
python script/train.py
```

This will initiate the training process using the model and dataset within the project. The scripts can be customized for further experimentation or optimization of the solution.

### Evaluation
Run a series of baseline and learned evaluations:
```bash
echo "Running evaluations..."
python script/eval_baselines_det.py
python script/eval_baselines_dyn.py
python script/eval_baselines_stoch.py
python script/eval_learned_det.py
python script/eval_learned_dyn.py
python script/eval_learned_stoch.py
```

### Visualization and Analysis
Generate learning curves, route visualizations, and summary statistics:
```bash
echo "Generating visualizations and analysis..."
python script/plot_learn_curves.py
python script/plot_routes.py
python script/results_to_tex.py
python script/routes_to_tex.py
```

### Testing
To run all test scripts, use the following loop:
```bash
echo "Running tests..."
for test_file in test/*.py
do
    echo "Running $test_file..."
    python "$test_file"
done
echo "All processes completed!"
```

## Monitoring and Debugging

- **Docker Logs**: To view container logs or troubleshoot issues:
  ```bash
  docker logs vrp_ambulance_container
  ```

- **Container Status**: Check the status of the container at any time:
  ```bash
  docker ps -a
  ```

## Project Structure

Below is a high-level overview of the project structure:

```
├── baseline/
│   ├── _base.py
│   ├── _critic.py
│   ├── _near_nb.py
│   ├── _no_bl.py
│   └── _rollout.py
├── cfgs/
│   └── gen_cfgs.py
├── docker/
│   └── Dockerfile            # Docker setup
├── externals/
│   ├── _lkh.py 
│   └── _ort.py
├── layers/
│   ├── _loss.py 
│   ├── _mha.py
│   └── _transformer.py
├── problems/
│   ├── _data.py              # Data handler for training
│   └── _env.py               # VRP environment definitions
├── script/
│   ├── train.py              # Main training script
│   ├── eval_baselines_det.py              
│   ├── eval_learned_det.py
│   ├── gen_val_data.py
│   ├── plot_learn_curves.py
│   ├── plot_routes.py
│   ├── results_to_tex.py
│   └── routes_to_tex.py
├── utils/ 
│   └── _args.py
└── README.md                 # Project documentation
```

### Key Files
- **Dockerfile**: Defines the Docker container environment.
- **train.py**: The primary script for training the VRP model.
- **eval_baselines_det.py**: Evaluation script for baseline deterministic models.
- **plot_learn_curves.py**: Script to generate learning curves and visualize training progress.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Added a new feature"`).
4. Push the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Please ensure your code adheres to the project’s style guidelines and includes relevant tests where necessary.

## License

This project is developed and maintained by **Hamideh Ahmadi, Hossein Afsharnia, and Maryam Pahlavan**.

