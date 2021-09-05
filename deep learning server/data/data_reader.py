import glob
import re
import random
import numpy as np
import cv2
from torch.utils.data import Dataset

from data.data_label_factory import label_factory

class DatasetReader(Dataset):
    def __init__(self, data):
        super(DatasetReader, self).__init__()
        self.data = [data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        videodata = self.data[idx]
        return videodata