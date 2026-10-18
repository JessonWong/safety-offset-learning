import os
import random
import numpy as np
from datasets import load_dataset


def _load_dataset(dataset_config):
    """Load dataset with optional subset."""
    if dataset_config.get('subset_name'):
        return load_dataset(dataset_config.dataset_name, dataset_config.subset_name)
    else:
        return load_dataset(dataset_config.dataset_name)


def prepare_dataset(dataset_config):
    dataset = _load_dataset(dataset_config)
    take_first_n = dataset_config.get('take_first_n', None)

    if take_first_n is not None:
        # Sequential split mode: use dataset order instead of random shuffling.
        main_split_name = dataset_config.get('main_split', 'train')
        main_dataset = dataset[main_split_name]
        total_size = len(main_dataset)
        train_size = min(int(take_first_n), total_size)
        val_size = min(dataset_config.get('validation_num', 0), max(0, total_size - train_size))
        test_size = min(dataset_config.get('test_num', 0), max(0, total_size - train_size - val_size))
        all_indices = list(range(total_size))
        splits_ids = {
            'train': all_indices[:train_size],
            'validation': all_indices[train_size:train_size + val_size],
            'test': all_indices[train_size + val_size:train_size + val_size + test_size],
        }
        print(
            f"Using sequential split: first {train_size} samples for train, "
            f"next {val_size} for validation, next {test_size} for test."
        )
    elif dataset_config.split_id_list_path is None or not os.path.exists(dataset_config.split_id_list_path):
        print(f"Split id list not found, creating a new one...")
        random.seed(dataset_config.random_seed)

        # Get the main split to create proper train/val/test splits
        main_split_name = dataset_config.get('main_split', 'train')
        main_dataset = dataset[main_split_name]
        total_size = len(main_dataset)

        # Create a single shuffled index list for the entire dataset
        all_indices = list(range(total_size))
        random.shuffle(all_indices)

        # Calculate split sizes
        train_size = dataset_config.get('train_num', 0)
        val_size = dataset_config.get('validation_num', 0)
        test_size = dataset_config.get('test_num', 0)

        # Create independent splits
        splits_ids = {}
        start_idx = 0

        if train_size > 0:
            splits_ids['train'] = all_indices[start_idx:start_idx + train_size]
            start_idx += train_size
        else:
            splits_ids['train'] = []

        if val_size > 0:
            splits_ids['validation'] = all_indices[start_idx:start_idx + val_size]
            start_idx += val_size
        else:
            splits_ids['validation'] = []

        if test_size > 0:
            splits_ids['test'] = all_indices[start_idx:start_idx + test_size]
        else:
            splits_ids['test'] = []
        if dataset_config.split_id_list_path is not None:
            print(f"Saving to {dataset_config.split_id_list_path}")
            os.makedirs(os.path.dirname(dataset_config.split_id_list_path), exist_ok=True)
            np.savez(dataset_config.split_id_list_path, **splits_ids)
    else:
        print(f"Split id list found, loading from {dataset_config.split_id_list_path}")
        splits_ids = np.load(dataset_config.split_id_list_path)
        splits_ids = {key: splits_ids[key] for key in splits_ids.files}

    # Create the actual splits using the main dataset
    main_split_name = dataset_config.get('main_split', 'train')
    splits = {key: dataset[main_split_name].select(splits_ids[key]) for key in splits_ids.keys()}
    splits['dataset'] = dataset
    return splits


def prepare_dataset_three_split(dataset_config):
    dataset = _load_dataset(dataset_config)

    if dataset_config.split_id_list_path is None or not os.path.exists(dataset_config.split_id_list_path):
        print(f"Split id list not found, creating a new one...")
        random.seed(dataset_config.random_seed)

        # Build indices from each split's own dataset (train/validation/test)
        splits_ids = {}
        desired_counts = {
            'train': dataset_config.get('train_num', 0),
            'validation': dataset_config.get('validation_num', 0),
            'test': dataset_config.get('test_num', 0),
        }
        for split_name, desired_count in desired_counts.items():
            split_size = len(dataset[split_name])
            indices = list(range(split_size))
            random.shuffle(indices)
            take_count = min(desired_count, split_size) if desired_count > 0 else 0
            splits_ids[split_name] = indices[:take_count]
        if dataset_config.split_id_list_path is not None:
            print(f"Saving to {dataset_config.split_id_list_path}")
            os.makedirs(os.path.dirname(dataset_config.split_id_list_path), exist_ok=True)
            np.savez(dataset_config.split_id_list_path, **splits_ids)
    else:
        print(f"Split id list found, loading from {dataset_config.split_id_list_path}")
        splits_ids = np.load(dataset_config.split_id_list_path)
        splits_ids = {key: splits_ids[key] for key in splits_ids.files}

    # Create the actual splits using each split's own dataset
    splits = {key: dataset[key].select(splits_ids[key]) for key in splits_ids.keys()}
    splits['dataset'] = dataset
    return splits


def prepare_dataset_train_validation_split(dataset_config):
    dataset = _load_dataset(dataset_config)
    train_split_name = dataset_config.get("train_split", "train")
    validation_split_name = dataset_config.get("validation_split", "validation")

    if train_split_name not in dataset:
        raise ValueError(f"Train split '{train_split_name}' not found in dataset {dataset_config.dataset_name}")
    if validation_split_name not in dataset:
        raise ValueError(
            f"Validation split '{validation_split_name}' not found in dataset {dataset_config.dataset_name}"
        )

    train_dataset = dataset[train_split_name]
    validation_dataset = dataset[validation_split_name]
    take_first_n = dataset_config.get("take_first_n", None)

    if take_first_n is not None:
        train_size = min(int(take_first_n), len(train_dataset))
        val_size = min(dataset_config.get("validation_num", 0), len(validation_dataset))
        test_size = min(
            dataset_config.get("test_num", 0),
            max(0, len(validation_dataset) - val_size),
        )
        splits_ids = {
            "train": list(range(train_size)),
            "validation": list(range(val_size)),
            "test": list(range(val_size, val_size + test_size)),
        }
        print(
            f"Using sequential split: first {train_size} samples from '{train_split_name}' for train, "
            f"first {val_size} samples from '{validation_split_name}' for validation, "
            f"next {test_size} samples from '{validation_split_name}' for test."
        )
    elif dataset_config.split_id_list_path is None or not os.path.exists(dataset_config.split_id_list_path):
        print("Split id list not found, creating a new one...")
        random.seed(dataset_config.random_seed)

        train_indices = list(range(len(train_dataset)))
        validation_indices = list(range(len(validation_dataset)))
        random.shuffle(train_indices)
        random.shuffle(validation_indices)

        train_size = dataset_config.get("train_num", 0)
        val_size = dataset_config.get("validation_num", 0)
        test_size = dataset_config.get("test_num", 0)

        splits_ids = {
            "train": train_indices[:train_size] if train_size > 0 else [],
            "validation": validation_indices[:val_size] if val_size > 0 else [],
            "test": validation_indices[val_size:val_size + test_size] if test_size > 0 else [],
        }
        if dataset_config.split_id_list_path is not None:
            print(f"Saving to {dataset_config.split_id_list_path}")
            os.makedirs(os.path.dirname(dataset_config.split_id_list_path), exist_ok=True)
            np.savez(dataset_config.split_id_list_path, **splits_ids)
    else:
        print(f"Split id list found, loading from {dataset_config.split_id_list_path}")
        splits_ids = np.load(dataset_config.split_id_list_path)
        splits_ids = {key: splits_ids[key] for key in splits_ids.files}

    splits = {
        "train": train_dataset.select(splits_ids.get("train", [])),
        "validation": validation_dataset.select(splits_ids.get("validation", [])),
        "test": validation_dataset.select(splits_ids.get("test", [])),
        "dataset": dataset,
    }
    return splits
