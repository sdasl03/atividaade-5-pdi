# src/data/dataloader.py
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
import torchio as tio  # Biblioteca excelente para imagens médicas

class BRATSDataset(torch.utils.data.Dataset):
    """Dataset real para BRATS"""
    
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mode = mode
        
        # Encontrar todos os pacientes
        self.patient_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        # Para BRATS, cada paciente tem:
        # - flair.nii.gz, t1.nii.gz, t1ce.nii.gz, t2.nii.gz (imagens)
        # - seg.nii.gz (máscara)
    
    def __len__(self):
        return len(self.patient_dirs)
    
    def load_nifti(self, path):
        """Carrega imagem NIfTI e normaliza"""
        img = nib.load(path).get_fdata()
        img = (img - img.mean()) / (img.std() + 1e-8)
        return img
    
    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        
        # Carregar modalidades
        modalities = ['flair', 't1', 't1ce', 't2']
        images = []
        
        for mod in modalities:
            mod_path = patient_dir / f"{mod}.nii.gz"
            img = self.load_nifti(mod_path)
            images.append(img)
        
        # Stack das modalidades [4, H, W, D]
        image = np.stack(images, axis=0)
        
        # Carregar máscara se disponível
        if self.mode != 'test':
            seg_path = patient_dir / "seg.nii.gz"
            mask = nib.load(seg_path).get_fdata()
            
            # BRATS tem 4 classes: 0=background, 1=necrosis, 2=edema, 3=enhancing
            # Podemos manter multiclasse ou converter para tarefa binária
            mask = mask.astype(np.int64)
        else:
            mask = np.zeros(image.shape[1:])
        
        # Converter para tensores
        image = torch.FloatTensor(image)
        mask = torch.LongTensor(mask)
        
        # Aplicar transformações
        if self.transform:
            # Para torchio, precisamos criar Subject
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                mask=tio.LabelMap(tensor=mask.unsqueeze(0))
            )
            subject = self.transform(subject)
            image = subject['image'].data
            mask = subject['mask'].data.squeeze(0)
        
        return image, mask

def get_transforms(mode='train'):
    """Transformações para dados médicos"""
    if mode == 'train':
        return tio.Compose([
            tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
            tio.RandomFlip(axes=(0, 1, 2)),
            tio.RandomNoise(std=0.01),
            tio.RandomBlur(std=(0, 0.5)),
            # Crop para tamanho fixo (ex: 128x128x128)
            tio.CropOrPad((128, 128, 128))
        ])
    else:
        return tio.Compose([
            tio.CropOrPad((128, 128, 128))
        ])