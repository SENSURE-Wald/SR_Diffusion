import argparse

import lightning as L

from benchmark.config_loader import SetupConfigLoader
from benchmark.data_preparator import DataPreparator
from benchmark.open_sr_strategy import OpenSRStrategy
from benchmark.swin2sr_strategy import Swin2SRStrategy


class Context:
    def __init__(self, strategy):
        self._strategy = strategy

    def execute_strategy(self):
        return self._strategy.execute()

def init_parser():
    parser = argparse.ArgumentParser(description="Run a benchmark with a specific model for super resolution.")
    parser.add_argument("-c", "--config", help="The configuration file for the benchmark setup.", type=str, required=False, default="opensr_model/configs/config_test_rgbi.yaml")
    parser.add_argument("-oc", "--opensr-config", help="The configuration file for the opensr model.", type=str, required=False, default="opensr_model/configs/config_10m.yaml")
    parser.add_argument("-sc", "--swin2sr-config", help="The configuration file for the Swin2SR model.", type=str, required=False, default="opensr_model/configs/config_transformer.yaml")
    return parser

# TODO: when adding new models
def choose_strategy(choice, data_loader, arguments, testing_config=None):
    match choice:
        case 1:
            return OpenSRStrategy(data_loader, arguments.opensr_config, testing_config)
        case 2:
            return Swin2SRStrategy(data_loader, arguments.swin2sr_config, testing_config)
        case _:
            raise ValueError("Invalid strategy choice.")

def main():
    argument_parser = init_parser()
    arguments = argument_parser.parse_args()

    # Load configuration
    print(f"[Benchmark] Loading setup config: {arguments.config}", flush=True)
    config_loader = SetupConfigLoader(arguments.config)
    print(f"[Benchmark] Setting random seed: {config_loader.testing_config.seed}", flush=True)
    L.seed_everything(config_loader.testing_config.seed, workers=True)

    # Prepare data
    print("[Benchmark] Preparing test data...", flush=True)
    data_preparator = DataPreparator(config_loader)
    data_loader = data_preparator.prepare_data()
    print(
        f"[Benchmark] Data ready: {len(data_loader.dataset)} samples, "
        f"{len(data_loader)} batches (batch size: {data_loader.batch_size}).",
        flush=True,
    )
    
    # Strategy handling
    for choice in [1,2]: # TODO: when adding new models
        model_name, config_path = {
            1: ("OpenSR", arguments.opensr_config),
            2: ("Swin2SR", arguments.swin2sr_config),
        }[choice]
        print(f"[Benchmark] Starting {model_name} with config: {config_path}", flush=True)
        strategy = choose_strategy(choice, data_loader, arguments, config_loader.testing_config)
        results = Context(strategy).execute_strategy()
        print(f"[Benchmark] {model_name} completed. Results: {results}", flush=True)

    print("[Benchmark] Benchmark completed.", flush=True)

if __name__ == "__main__":
    main()
