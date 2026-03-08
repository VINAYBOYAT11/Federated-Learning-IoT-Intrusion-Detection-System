<div align="center">

# 🌐 Federated Learning Based IoT Intrusion Detection System

### *Minor Project — Semester 6*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower_FL-Federated_Learning-purple?style=for-the-badge)](https://flower.dev)
[![MQTT](https://img.shields.io/badge/Protocol-MQTT-660066?style=for-the-badge&logo=eclipse-mosquitto&logoColor=white)]()
[![Wireshark](https://img.shields.io/badge/Tool-Wireshark-1679A7?style=for-the-badge&logo=wireshark&logoColor=white)](https://wireshark.org)

> 🔐 A privacy-preserving intrusion detection system for IoT networks using **Federated Learning** (Flower framework) and **Convolutional Neural Networks** — without sharing raw data across nodes.

</div>

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Motivation](#-motivation)
- [System Architecture](#-system-architecture)
- [How Federated Learning Works](#-how-federated-learning-works)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [References](#-references)
- [Author](#-author)

---

## 📄 Abstract

The rapid growth of **Internet of Things (IoT)** devices has significantly expanded the attack surface for network intrusions. Traditional centralized Intrusion Detection Systems (IDS) require raw data to be sent to a central server — violating **privacy** and creating **single points of failure**.

This project implements a **Federated Learning-based IDS** using the **Flower (flwr)** framework, where:
- Multiple IoT clients **train locally** on their own data
- Only **model weights** (not raw data) are shared with the central server
- The server **aggregates** the weights using **FedAvg** strategy
- The global model improves **round by round** without any data leaving the device

The system is evaluated on the **MQTT-IoT-IDS2020 dataset** — a real-world IoT traffic dataset.

---

## 💡 Motivation

| Problem | Our Solution |
|---------|-------------|
| IoT devices generate sensitive traffic data | **Federated Learning** — data never leaves the device |
| Centralized IDS = single point of failure | **Distributed training** across multiple IoT nodes |
| High communication overhead | **Only gradients/weights** are transmitted |
| Privacy regulations (GDPR etc.) | **Local training** ensures raw data privacy |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING                     │
│                                                          │
│   ┌──────────┐     ┌──────────────────┐    ┌──────────┐ │
│   │ Client 1 │────▶│                  │◀───│ Client N │ │
│   │ IoT Data │     │   FL SERVER      │    │ IoT Data │ │
│   │  + CNN   │◀────│  (FedAvg Aggr.)  │───▶│  + CNN   │ │
│   └──────────┘     │                  │    └──────────┘ │
│                    │  Global Model    │                  │
│   ┌──────────┐────▶│  Evaluation      │◀───┌──────────┐ │
│   │ Client 2 │     │                  │    │ Client M │ │
│   │ IoT Data │◀────│  Hydra Config    │───▶│ IoT Data │ │
│   └──────────┘     └──────────────────┘    └──────────┘ │
└──────────────────────────────────────────────────────────┘
```

**Flow:**
1. Server sends global model to selected clients
2. Each client trains on **local IoT traffic data**
3. Clients send back only **updated model weights**
4. Server aggregates updates using **FedAvg**
5. Repeat for `N` rounds → improved global intrusion detector

---

## 🔄 How Federated Learning Works

```
Round 1:  Server → Sends initial model weights to clients
          Clients → Train locally on MQTT IoT traffic data
          Clients → Send updated weights back (NOT raw data)
          Server  → FedAvg aggregation → New global model

Round 2:  [Repeat with improved model]
   .
   .
Round N:  Global model converges → Final IDS deployed
```

The **FedAvg (Federated Averaging)** algorithm:

```
W_global = Σ (n_k / n) × W_k
```
Where `W_k` = weights from client `k`, `n_k` = number of samples at client `k`, `n` = total samples.

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Core language |
| **PyTorch** | Latest | Neural network (CNN) training |
| **Flower (flwr)** | Latest | Federated Learning orchestration |
| **Hydra** | Latest | Configuration management |
| **OmegaConf** | Latest | YAML config parsing |
| **Wireshark** | 4.2.4 | Network traffic capture & analysis |
| **MQTT Protocol** | — | IoT communication protocol for dataset |

---

## 📊 Dataset

**MQTT-IoT-IDS2020** — A benchmark dataset for IoT intrusion detection

| Property | Details |
|----------|---------|
| **Protocol** | MQTT (Message Queuing Telemetry Transport) |
| **Traffic Types** | Normal, DoS, Reconnaissance, MITM, Injection |
| **Format** | Network traffic captures (PCAP → features) |
| **Use Case** | Multi-class IoT intrusion classification |
| **Source** | [Research Paper Reference](https://github.com/salmanjabr/MQTT-IoT-IDS2020) |

---

## 📁 Project Structure

```
flower/  (minor_project_sem6)
│
├── 📄 main.py              # Entry point — FL simulation orchestrator
├── 📄 client.py            # FlowerClient — local training logic
├── 📄 server.py            # Server-side evaluation & config functions
├── 📄 model.py             # CNN model (PyTorch) — Net class
├── 📄 dataset.py           # Dataset loading & IID partitioning
│
├── 📁 conf/
│   └── base.yaml           # Hydra config (rounds, clients, lr, etc.)
│
├── 📁 outputs/             # Experiment results (auto-generated by Hydra)
├── 📁 experiment/          # Experimental runs & logs
├── 📁 midterm/             # Midterm submission materials
├── 📁 endterm/             # Endterm submission materials
├── 📁 test/                # Test scripts
├── 📁 uploade/             # Upload logs / artifacts
│
├── 📄 Project-Report-Vinay.pdf    # Full project report
└── 📄 README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
python --version    # Python 3.8+
pip --version
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/VINAYBOYAT11/flower.git
cd flower

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate         # Windows
source venv/bin/activate      # Linux/Mac

# 3. Install dependencies
pip install flwr torch torchvision hydra-core omegaconf

# 4. Run the Federated Learning simulation
python main.py
```

### Configuration

Edit `conf/base.yaml` to customize:

```yaml
num_clients: 10              # Total federated clients
num_rounds: 5                # FL training rounds
batch_size: 32
num_classes: 10

num_clients_per_round_fit: 5   # Clients selected per round
num_clients_per_round_eval: 5

config_fit:
  lr: 0.01
  momentum: 0.9
  local_epochs: 1
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| **Model** | CNN (Conv2D → Pool → FC layers) |
| **Strategy** | FedAvg |
| **Rounds** | 5 FL rounds |
| **Accuracy** | *See project report* |
| **Loss** | *Cross-Entropy Loss* |

Results are saved automatically to `outputs/<date>/<time>/results.pkl` by Hydra.

---

## 📚 References

1. **Flower: A Friendly Federated Learning Framework** — [paper](https://arxiv.org/abs/2007.14390)
2. **MQTT-IoT-IDS2020 Dataset** — *Machine Learning Based IoT Intrusion Detection System: An MQTT Case Study*
3. **FedAvg Algorithm** — McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, 2017
4. **PyTorch Documentation** — [pytorch.org](https://pytorch.org)

---

## 👨‍💻 Author

**Vinay Boyat**

*Minor Project — B.E. Computer Science, Semester 6*

[![GitHub](https://img.shields.io/badge/GitHub-VINAYBOYAT11-181717?style=for-the-badge&logo=github)](https://github.com/VINAYBOYAT11)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live-00C7B7?style=for-the-badge&logo=vercel)](https://portfolio-pink-beta-1y6l6m0ylt.vercel.app)

---

<div align="center">

> 🔐 *"Privacy is not something that I'm merely entitled to, it's an absolute prerequisite." — Marlon Brando*
>
> *Applied to IoT — your traffic data stays yours, your model still learns.*

⭐ **Found this useful? Star the repo!** ⭐

*Made with ❤️ by Vinay Boyat — Minor Project Sem 6*

</div>
