# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.utils.data
from loaders import image_folder


class DAVISDataLoader():
    def __init__(self, list_path, _batch_size):
        dataset = image_folder.DAVISImageFolder(list_path=list_path)
        self.data_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=_batch_size,
                                                       shuffle=False,
                                                       num_workers=int(1)) #TODO add sampler = ... here
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'TestDataLoader'

    def __len__(self):
        return len(self.dataset)

class DAVISFourierDataLoader():
    def __init__(self, list_path, _batch_size):
        dataset = image_folder.DAVISFourierImageFolder(list_path=list_path)
        self.data_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=_batch_size,
                                                       shuffle=False,
                                                       num_workers=int(1)) #TODO add sampler = ... here
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'TestFourierDataLoader'

    def __len__(self):
        return len(self.dataset)


class TUMDataLoader():
    def __init__(self, opt, list_path, is_train, _batch_size, num_threads):
        dataset = image_folder.TUMImageFolder(opt=opt, list_path=list_path)
        self.data_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=_batch_size,
                                                       shuffle=False,
                                                       num_workers=int(num_threads))
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'TUMDataLoader'

    def __len__(self):
        return len(self.dataset)
        
class SupervisionDataLoader():
    def __init__(self, list_path, _batch_size):
        dataset = image_folder.SupervisionImageFolder(list_path=list_path)
        self.data_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=_batch_size,
                                                       shuffle=True,
                                                       num_workers=int(1))
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'SupervisionDataLoader'

    def __len__(self):
        return len(self.dataset)

class SupervisionFourierDataLoader():
    def __init__(self, list_path, _batch_size):
        dataset = image_folder.SupervisionFourierImageFolder(list_path=list_path)
        self.data_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=_batch_size,
                                                       shuffle=True,
                                                       num_workers=int(1))
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'SupervisionFourierDataLoader'

    def __len__(self):
        return len(self.dataset)

# NOTE: For Supervision + Latent Constraint Loss
class SupervisionLatentDataLoader():
    def __init__(self, list_path, _batch_size):
        dataset = image_folder.SupervisionLatentImageFolder(list_path=list_path)
        self.data_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=_batch_size,
                                                       shuffle=True,
                                                       num_workers=int(1))
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'SupervisionLatentDataLoader'

    def __len__(self):
        return len(self.dataset)