import os

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ShapeNetDINO(Dataset):
    def __init__(self, 
                 data_root, 
                 split, 
                 categories=None, 
                 pc_dir='pointcloud', 
                 occupancy_dir='occupancy', 
                 num_pts=2048,
                 num_samples=2048, 
                 transform=None):
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.shuffle = (split == "train")
        
        self.num_pts = num_pts
        self.num_samples = num_samples
        self.transform = transform

        if categories is None:
            categories = os.listdir(self.data_root)
            categories = [c for c in categories if os.path.isdir(os.path.join(self.data_root, c)) and c.startswith('0')]
        categories.sort()
        print(categories)

        self.pc_dir = pc_dir
        self.occupancy_dir = occupancy_dir

        self.shape_models = []
        self.category_ids = {}
        for cat_idx, cat in enumerate(categories):
            self.category_ids[cat] = cat_idx
            subpath = os.path.join(self.data_root, cat)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_cat = f.read().split('\n')
            
            self.shape_models += [
                {'category': cat, 'model': m.split('.')[0]}
                for m in models_cat
            ]

    def __getitem__(self, idx):
        category = self.models[idx]['category']
        model = self.models[idx]['model']

        occupancy_path = os.path.join(self.data_root, category, self.occupancy_dir, model+'.npz')
        with open(occupancy_path.replace('.npz', '.npy'), 'rb') as f:
            scale = np.load(f).item()

        pc_path = os.path.join(self.data_root, category, self.pc_dir, model+'.npz')
        with np.load(pc_path) as data:
            surface = data['points'].astype(np.float32)
            surface = surface * scale

        ind = np.random.default_rng().choice(surface.shape[0], self.num_pts, replace=False)
        surface = surface[ind]
        surface = torch.from_numpy(surface)

        if self.transform:
            surface = self.transform(surface)

        return surface, self.category_ids[category]
        

    def __len__(self):
        return len(self.shape_models)


class ShapeNet3DILG(Dataset):
    def __init__(self, 
                 data_root, 
                 split, 
                 categories=None, 
                 pc_dir='pointcloud', 
                 occupancy_dir='occupancy', 
                 num_pts=2048,
                 num_samples=2048, 
                 transform=None, 
                 return_surface=True):
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.shuffle = (split == "train")
        
        self.num_pts = num_pts
        self.num_samples = num_samples
        self.transform = transform
        self.return_surface = return_surface

        if categories is None:
            categories = os.listdir(self.data_root)
            categories = [c for c in categories if os.path.isdir(os.path.join(self.data_root, c)) and c.startswith('0')]
        categories.sort()
        print(categories)

        self.pc_dir = pc_dir
        self.occupancy_dir = occupancy_dir

        self.shape_models = []
        self.category_ids = {}
        for cat_idx, cat in enumerate(categories):
            self.category_ids[cat] = cat_idx
            subpath = os.path.join(self.data_root, cat)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_cat = f.read().split('\n')
            
            self.shape_models += [
                {'category': cat, 'model': m.split('.')[0]}
                for m in models_cat
            ]

    def __getitem__(self, idx):
        category = self.models[idx]['category']
        model = self.models[idx]['model']

        occupancy_path = os.path.join(self.data_root, category, self.occupancy_dir, model+'.npz')
        with open(occupancy_path.replace('.npz', '.npy'), 'rb') as f:
            scale = np.load(f).item()

        if self.return_surface:
            pc_path = os.path.join(self.data_root, category, self.pc_dir, model+'.npz')
            with np.load(pc_path) as data:
                surface = data['points'].astype(np.float32)
                surface = surface * scale

            ind = np.random.default_rng().choice(surface.shape[0], self.num_pts, replace=False)
            surface = surface[ind]
            surface = torch.from_numpy(surface)

        try:
            with np.load(occupancy_path) as data:
                vol_points = data['vol_points']
                vol_label = data['vol_label']
                near_points = data['near_points']
                near_label = data['near_label']
        except Exception as e:
            print(e)
            print(occupancy_path)

        ind_vol = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
        vol_points = vol_points[ind_vol]
        vol_label = vol_label[ind_vol]
        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()

        if self.shuffle:
            ind_near = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
            near_points = near_points[ind_near]
            near_label = near_label[ind_near]
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label

        if self.transform:
            surface, points = self.transform(surface, points)

        if self.return_surface:
            return surface, points, labels, self.category_ids[category]
        else:
            return points, labels, self.category_ids[category]
        

    def __len__(self):
        return len(self.shape_models)
    

if __name__ == '__main__':
    pass
