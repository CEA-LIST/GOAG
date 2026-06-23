import lightning as L
from torch.utils.data import DataLoader

from utils_data.BPSCustomDataset import BPSCustomDataset

class DataModule(L.LightningDataModule):
    def __init__(self, cfg, shuffle=False):
        super().__init__()
        self.cfg = cfg
        self.shuffle = shuffle
    
    def setup(self, stage=''):
        
        print("Loading dataset... ")
        self.dataset = BPSCustomDataset(robot_name=self.cfg['robot_name'], normalize_jv=self.cfg['normalize_joint_values'], split=True, subset=False)
        print("Dataset Loaded !")

        self.train_dataset = self.dataset.train_dataset
        self.validate_dataset = self.dataset.test_dataset

        print("[DataModule] Dataset size  : ", self.dataset.__len__())
        print("[DataModule]     Train     : ", self.train_dataset.__len__())
        print("[DataModule]     Validate  : ", self.validate_dataset.__len__())
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg['batch_size'], 
            shuffle=self.shuffle, 
            num_workers=self.cfg['num_workers'], 
            collate_fn=self.train_dataset.__collate_fn__, 
            multiprocessing_context='fork' if self.cfg['num_workers'] > 0 else None, 
            persistent_workers=True if self.cfg['num_workers'] > 0 else False)

    def val_dataloader(self):
        return DataLoader(
            self.validate_dataset, 
            batch_size=self.cfg['batch_size'], 
            shuffle=self.shuffle, 
            num_workers=self.cfg['num_workers'], 
            collate_fn=self.validate_dataset.__collate_fn__, 
            multiprocessing_context='fork' if self.cfg['num_workers'] > 0 else None, 
            persistent_workers=True if self.cfg['num_workers'] > 0 else False)

    def full_dataloader(self):
        return DataLoader(
            self.dataset, 
            batch_size=self.cfg['batch_size'], 
            shuffle=self.shuffle, 
            num_workers=self.cfg['num_workers'], 
            collate_fn=self.dataset.__collate_fn__, 
            multiprocessing_context='fork' if self.cfg['num_workers'] > 0 else None, 
            persistent_workers=True if self.cfg['num_workers'] > 0 else False)
    
