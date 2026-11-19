import argparse

from benchmark.config_loader import SetupConfigLoader
from benchmark.data_preparator import DataPreparator
from benchmark.open_sr_strategy import OpenSRStrategy
from benchmark.swin2sr_strategy import Swin2SRStrategy


class Context:
    def __init__(self, strategy):
        self._strategy = strategy

    def execute_strategy(self):
        self._strategy.execute()

def init_parser():
    parser = argparse.ArgumentParser(description="Run a benchmark with a specific model for super resolution.")
    parser.add_argument("-c", "--config", help="The configuration file for the benchmark setup.", type=str, required=False, default="opensr_model/configs/config_test_rgbi.yaml")
    parser.add_argument("-oc", "--opensr-config", help="The configuration file for the opensr model.", type=str, required=False, default="opensr_model/configs/config_10m.yaml")
    parser.add_argument("-sc", "--swin2sr-config", help="The configuration file for the swin2sr model.", type=str, required=False, default="opensr_model/configs/config_transformer.yaml")
    return parser

# TODO: when adding new models
def choose_strategy(choice, data_loader, arguments):
    match choice:
        case 1:
            return OpenSRStrategy(data_loader, arguments.opensr_config)
        case 2:
            return Swin2SRStrategy(data_loader, arguments.swin2sr_config)
        case _:
            raise ValueError("Invalid strategy choice.")

def main():
    argument_parser = init_parser()
    arguments = argument_parser.parse_args()

    # Load configuration
    config_loader = SetupConfigLoader(arguments.config)

    # Prepare data
    data_preparator = DataPreparator(config_loader)
    data_loader = data_preparator.prepare_data()
    
    # Strategy handling
    for choice in [1,2]: # TODO: when adding new models
        strategy = choose_strategy(choice, data_loader, arguments)
        Context(strategy).execute_strategy()

    print ("Benchmark completed.")

if __name__ == "__main__":
    main()