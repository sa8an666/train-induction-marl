 Multi-Agent Train Induction RL

This project implements a **simple Multi-Agent Reinforcement Learning (MARL)** environment simulating train induction on a single platform. It compares the performance of **DQN** and **PPO** agents.

---

## 📚 Project Overview

- **Environment:** Multiple train approaches trying to access a single platform.
- **Agents:** Each approach is an agent deciding whether to **WAIT (0)** or **INDUCT (1)**.
- **Safety rule:** Only one train can occupy the platform at a time; collisions are heavily penalized.
- **Goal:** Minimize queue delays and avoid collisions while maximizing successful train inductions.

---

## 🛠 Features

- **DQN agents**: Trained independently for each approach using a replay buffer and target network.
- **PPO agents**: Trained with policy gradients using advantages and clipped objective.
- **Episode rewards plotting**: Compares DQN vs PPO performance over training episodes.
- **Model saving**: Trained policy networks are saved in `saved_models/`.

---

## 📂 Repository Structure

train-induction-marl/
│
├── train_marl_ppo_plot.py # Main script: environment, DQN & PPO agents, training & plotting
├── requirements.txt # Python dependencies
├── README.md # Project description
├── .gitignore # Git ignore rules
├── saved_models/ # Trained models (auto-saved)
└── plots/ # Episode reward plots (auto-saved)


> `venv/` is not included in the repo; create your own virtual environment.

---

## ⚡ Installation & Usage

1. **Clone the repository**

```bash
git clone https://github.com/sa8an666/train-induction-marl.git
cd train-induction-marl


2. Create a virtual environment

python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows


3. Install dependencies

pip install -r requirements.txt


4. Run the training script

python train_marl_ppo_plot.py


This trains both DQN and PPO agents and saves a comparison plot in plots/dqn_vs_ppo.png.

Trained models are saved in saved_models/.

📊 Results
After training, you can visualize the total episode rewards for both DQN and PPO.

The script automatically generates a plot showing learning performance and stability of each method.

📝 Notes
Increase the number of episodes for more stable PPO training.

The environment is intentionally simple for understanding and experimentation.

Modify hyperparameters in train_marl_ppo_plot.py to explore different behaviors.
