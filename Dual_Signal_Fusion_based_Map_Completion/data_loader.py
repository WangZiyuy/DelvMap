import torch.utils.data as data
import numpy as np
import os
import torchvision.transforms as transforms
import cv2


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float")
    return img


def get_data_loader_multistage(root_dir, mode):
    if mode == 'train':
        dl = data.DataLoader(MultistageDataset(os.path.join(root_dir, mode)), shuffle=True, batch_size=1)
    else:
        dl = data.DataLoader(MultistageDataset(os.path.join(root_dir, mode)), shuffle=True, batch_size=1)
    return dl


class MultistageDataset(data.Dataset):
    def __init__(self, data_path):
        super(MultistageDataset, self).__init__()
        # main task
        self.traj_paths = [os.path.join(data_path, 'traj_and_point_split', filename) for filename in os.listdir(os.path.join(data_path, 'traj_and_point_split'))
                     if filename.endswith('.npy')]
        self.label_paths = [os.path.join(data_path, 'label', filename) for filename in os.listdir(os.path.join(data_path, 'label'))
                       if filename.endswith('.png')]
        # building task
        self.src_paths = [os.path.join(data_path, 'src_split', filename) for filename in os.listdir(os.path.join(data_path, 'src_split'))
                     if filename.endswith('.png')]
        self.building_label_paths = [os.path.join(data_path, 'building_label', filename) for filename in os.listdir(os.path.join(data_path, 'building_label'))
                     if filename.endswith('.png')]
        # for test
        self.image_paths = [os.path.join(data_path, 'traj_and_point_split', filename) for filename in os.listdir(os.path.join(data_path, 'traj_and_point_split'))
                     if filename.endswith('.npy')]

    def __getitem__(self, index):
        traj_path = self.traj_paths[index]
        label_path = self.label_paths[index]
        src_path = self.src_paths[index]
        building_label_path = self.building_label_paths[index]
        img_path = self.image_paths[index]

        traj = np.asarray(np.load(traj_path))
        traj = np.array(traj, dtype="float")
        label = load_img(str(label_path), grayscale=True)
        label = np.expand_dims(label, axis=-1)

        src = load_img(src_path, grayscale=False)
        building_label = load_img(str(building_label_path), grayscale=True)
        building_label = np.expand_dims(building_label, axis=-1)

        img_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        img = img_transform(np.array(traj, dtype="uint8")).float()
        label = img_transform(label * 128).float()
        src = img_transform(np.array(src, dtype="uint8")).float()
        building_label = img_transform(building_label * 255).float()

        return {
            'traj_path': traj_path,
            'traj_data': img,
            'label_path': label_path,
            'label_data': label,
            'src_path': src_path,
            'src_data': src,
            'building_label_path': building_label_path,
            'building_label_data': building_label,

            'img_path': img_path,

        }

    def __len__(self):
        return len(self.traj_paths)
