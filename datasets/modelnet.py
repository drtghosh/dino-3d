import os

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from datasets.data_utils import read_point_cloud_off

from torch_cluster import fps, knn


def get_dataloader_modelnet40(split, config):
    is_shuffle = (split == 'train')
    if "dino" in config.module:
        dataset = ModelNet40DINO(split, config.data_root, config.n_pts, config.n_local, config.local_pts)
    else:
        raise ValueError("Unknown module: {}".format(config.module))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    
    return dataloader


def split_data(path, split):
    # find all categories (alphabetically sorted)
    classes = sorted(os.listdir(path))

    # create category id dictionary
    cat2id = dict()
    for idx, cls in enumerate(classes):
        cat2id[cls] = idx

    # collect all splitwise paths
    split_paths = list()
    split_labels = list()
    for cls in classes:
        class_path = os.path.join(path, cls)
        split_path = os.path.join(class_path, split)
        class_split_files = sorted(os.listdir(split_path))
        class_split_paths = [os.path.join(cls, split, tf) for tf in class_split_files]
        split_paths.extend(class_split_paths)
        split_labels.extend(len(class_split_paths) * [cat2id[cls]])

    return split_paths, split_labels


def split_data_with_classes(path, split):
    # find all categories (alphabetically sorted)
    classes = sorted(os.listdir(path))

    # create category id dictionary
    cat2id = dict()
    for idx, cls in enumerate(classes):
        cat2id[cls] = idx

    # collect all splitwise paths
    split_paths = list()
    split_labels = list()
    for cls in classes:
        class_path = os.path.join(path, cls)
        split_path = os.path.join(class_path, split)
        class_split_files = sorted(os.listdir(split_path))
        class_split_paths = [os.path.join(cls, split, tf) for tf in class_split_files]
        split_paths.extend(class_split_paths)
        split_labels.extend(len(class_split_paths) * [cat2id[cls]])

    return split_paths, split_labels, classes


def split_data_without_labels(path, split):
    # find all categories (alphabetically sorted)
    classes = sorted(os.listdir(path))

    # collect all splitwise paths
    split_paths = list()
    for cls in classes:
        class_path = os.path.join(path, cls)
        split_path = os.path.join(class_path, split)
        class_split_files = sorted(os.listdir(split_path))
        class_split_paths = [os.path.join(cls, split, tf) for tf in class_split_files]
        split_paths.extend(class_split_paths)

    return split_paths, classes


class ModelNet40DINO(Dataset):
    def __init__(self, split, data_root, n_pts, n_local, local_pts):
        super(ModelNet40DINO, self).__init__()
        self.split = split
        self.shuffle = (split == "train")
        self.data_root = data_root
        self.paths, self.classes = split_data(data_root, split)
        self.n_pts = n_pts
        self.n_local = n_local
        self.local_pts = local_pts

    def __getitem__(self, index):
        # read point cloud
        pc_path = os.path.join(self.data_root, self.paths[index])
        pc = read_point_cloud_off(pc_path)

        # normalize point cloud
        assert len(pc.shape) == 2
        norm_pc = pc - np.mean(pc, axis=0)
        norm_pc /= np.max(np.linalg.norm(norm_pc, axis=1))

        # create global view
        tensor_pc = torch.tensor(norm_pc, dtype=torch.float32)
        fps_idx_global = fps(tensor_pc, ratio=self.n_pts/len(norm_pc))
        global_points = tensor_pc[fps_idx_global]

        fps_idx_local = fps(global_points, ratio=self.n_local/self.n_pts)
        row, col = knn(global_points, global_points[fps_idx_local], self.local_pts)
        local_points = global_points[col]
        local_points = local_points.view(self.n_local, -1, 3)

        return {"id": self.paths[index], "global_view": global_points, "local_views": local_points}

    def __len__(self):
        return len(self.paths)


if __name__ == "__main__":
    pass