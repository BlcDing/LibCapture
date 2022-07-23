import flwr as fl
from typing import List, Optional


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List,
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(
            f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}"
        )

        # Call aggregate_evaluate from base class (FedAvg)
        params, _ = super().aggregate_evaluate(rnd, results, failures)
        return params, {"accuracy": accuracy_aggregated}


# Define strategy
strategy = AggregateCustomMetricStrategy()

# Start server
fl.server.start_server(
    server_address="[::]:9999",
    config={"num_rounds": 200},
    strategy=strategy,
)
