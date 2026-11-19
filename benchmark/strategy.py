class Strategy:
    """Base class for all strategies."""

    def __init__(self, data_loader, config_path):
        self.data_loader = data_loader
        self.config_path = config_path