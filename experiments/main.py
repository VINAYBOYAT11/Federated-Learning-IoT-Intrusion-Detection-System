# main.py

import pickle
from pathlib import Path
from typing import Dict

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from client import MQTTIDSClient
from server import MQTTIDSServer
from dataset import prepare_mqtt_iot_ids2020_dataset
from model import MQTTIDSModel


@hydra.main(config_path="conf", config_name="base", strict=False)
def main(cfg: DictConfig):
    # Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    save_path = Path(HydraConfig.get().run.dir)

    # Prepare MQTT-IoT-IDS2020 dataset
    trainloaders, valloaders, testloader = prepare_mqtt_iot_ids2020_dataset(cfg.batch_size)

    # Define MQTT IDS model
    model = MQTTIDSModel(num_classes=2)  # Assuming binary classification

    # Define server
    server = MQTTIDSServer(
        model=model,
        num_rounds=cfg.num_rounds,
        num_clients=cfg.num_clients,
        config_fit=cfg.config_fit,
        evaluation_metrics=["accuracy", "precision", "recall", "f1-score"]
    )

    # Define clients
    clients = []
    for trainloader, valloader in zip(trainloaders, valloaders):
        client = MQTTIDSClient(trainloader=trainloader, valloader=valloader)
        clients.append(client)

    # Start simulation
    fl.simulation.start_simulation(
        server=server,
        clients=clients
    )

    # Save results
    results_path = save_path / "results.pkl"
    results = {"history": server.history}
    with open(results_path, "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
