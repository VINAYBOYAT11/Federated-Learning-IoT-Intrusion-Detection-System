# server.py

from typing import Dict

import torch
from flwr.server import Server
from model import MQTTIDSModel


class MQTTIDSServer(Server):
    def __init__(self, model: MQTTIDSModel, num_rounds: int, num_clients: int, config_fit: Dict[str, float], evaluation_metrics: List[str]) -> None:
        super().__init__()
        self.model = model
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.config_fit = config_fit
        self.evaluation_metrics = evaluation_metrics

    def get_on_fit_config_fn(self):
        def on_fit_config_fn(round_num: int):
            return self.config_fit

        return on_fit_config_fn

    def evaluate_fn(self, parameters: List[NDArray], config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        # Aggregate model parameters
        aggregated_parameters = self.aggregate(parameters)

        # Update global model with aggregated parameters
        self.model.set_parameters(aggregated_parameters)

        # Evaluate global model
        evaluation_results = {}
        for metric in self.evaluation_metrics:
            evaluation_results[metric] = self.compute_metric(metric)

        return evaluation_results
