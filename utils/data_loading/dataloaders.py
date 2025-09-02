from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from utils.data_modification.collate_data import DataCollater
from functools import partial


def create_test_dataloaders(data):
    """
    creates test dataloaders
    Args:
        data: dict of data parameters

    Returns:
        test_loaders: either None or a dict of dataloaders per study and per patient

    """
    # create test set dataloader if test set is not None
    test_loaders = {}
    for type in data.keys():
        test_loaders[type] = {}
        for application in data[type].keys():
            test_loaders[type][application] = {}
            for study in data[type][application].keys():
                test_loaders[type][application][study] = {}
                for patient_id, patient in enumerate(data[type][application][study]):
                    test_loader = DataLoader(patient, batch_size=1, shuffle=False,
                                             collate_fn=DataCollater.collate_data, drop_last=False)
                    test_loaders[type][application][study][patient_id] = test_loader
    return test_loaders


def create_train_dataloaders(train_set, batch_size, steps_per_epoch, qmri_model, weights, weighted=False):
    """
    creates train dataloaders
    Args:
        train_set: either None or a dict
        batch_size: int, the batch size
        steps_per_epoch: int, the number of steps per epoch
        qmri_model: a qmri model

    Returns:
        train_loader: either None or a dict of dataloaders per study and per patient
    """

    # create train set dataloader
    if weighted:
        train_sampler = WeightedRandomSampler(weights, replacement=True, num_samples=batch_size*steps_per_epoch)
    else:
        train_sampler = RandomSampler(train_set, replacement=True, num_samples=batch_size*steps_per_epoch)
    data_collater = DataCollater(qmri_model=qmri_model)
    train_loader = DataLoader(dataset=train_set, sampler=train_sampler,
                              collate_fn=data_collater.random_collate_data, batch_size=batch_size, num_workers=2)
    return train_loader


def create_val_dataloaders(val_set, batch_size, qmri_model):
    """
    creates val dataloaders
    Args:
        val_set: either None or a dict
        batch_size: int, the batch size
        qmri_model: a qmri model

    Returns:
        val_loader: either None or a dict of dataloaders per study and per patient
    """

    # create val set dataloader if val set is not None
    data_collater = DataCollater(qmri_model=qmri_model)
    if val_set is not None:
        val_sampler = RandomSampler(val_set, replacement=True, num_samples=batch_size)
        val_loader = DataLoader(dataset=val_set, sampler=val_sampler,
                                collate_fn=data_collater.random_collate_data, batch_size=batch_size, num_workers=2)
    else:
        val_loader = None

    return val_loader
