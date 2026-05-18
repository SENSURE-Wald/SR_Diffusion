from torch.utils.data import Dataset
import tacoreader
import pathlib


class SEN2NAIPDataset(Dataset):

    def __init__(self, dataset_config, split: str = "train"):
        super().__init__()
        self.repo_id = dataset_config.repo_id
        self.file_name = dataset_config.file_name
        self.local_dir = pathlib.Path(dataset_config.local_dir)
        self.split = split
        dataset = tacoreader.load("tacofoundation:sen2naipv2-crosssensor")
        print(dataset[dataset["tortilla:data_split"] != "train"])
