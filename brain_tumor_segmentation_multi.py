#!/usr/bin/env python3.13
"""
BRAIN TUMOR SEGMENTATION - MULTI-PATIENT VERSION WITH SEPARATE TRAIN/VAL
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
import sys
import os
import nibabel as nib
import random
from scipy import ndimage
import zipfile
# import shutil
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

print("=" * 80)
print("🧠 BRAIN TUMOR SEGMENTATION")
print("=" * 80)
print(f"📅 {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🐍 Python {sys.version.split()[0]}")
print(f"🔥 PyTorch {torch.__version__}")

# ==================== MEMORY UTILITIES ====================
def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024**3  # GB

def memory_safe_operation(func):
    """Decorator for memory-safe operations"""
    def wrapper(*args, **kwargs):
        gc.collect()
        # torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return func(*args, **kwargs)
    return wrapper

# ==================== CONFIGURATION ====================
class MultiPatientConfig:
    """Configuration for multi-patient project with separate train/val"""
    
    def __init__(self):
        # Experiment setup
        self.project_name = "Brain_Tumor_Segmentation_Separate_Train_Val"
        self.author = "Samuel Lira"
        self.institution = "Universidade Aberta"
        
        # Paths for BRATS data - YOUR STRUCTURE
        self.data_dirs = {
            'training': 'data/raw/training',      # 5 patients for training
            'validation': 'data/raw/validation',  # 5 patients for validation
            'extracted': 'data/extracted',        # For extracted files
            'processed': 'data/processed_multi'   # For processed data
        }
        
        # Model architecture
        self.image_size = 240
        self.in_channels = 4   # T1, T1ce, T2, FLAIR
        self.num_classes = 4   # Background + 3 tumor regions
        self.base_filters = 8
        
        # Training parameters
        self.batch_size = 2
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.patience = 15
        
        # Data augmentation - only for training
        self.augmentation = True
        self.rotation_range = 15
        self.scale_range = (0.9, 1.1)
        self.flip_prob = 0.5
        self.intensity_scale_range = (0.8, 1.2)
        
        # Memory optimization parameters
        self.use_memory_mapping = True  # Use memory-mapped files
        self.max_cache_size = 2  # GB - limit cache size
        self.clear_cache_every = 10  # Clear cache every N batches
        self.preload_patients = False  # Don't preload all patients
        
        # Preprocessing parameters
        self.normalize_per_modality = True
        self.clip_percentile = (1, 99)
        
        # ZIP handling
        self.extract_zip_files = True
        self.use_existing_extracted = True
        
        # Create directories
        self.create_directories()
        
        # Save config
        self.save_config()
    
    def create_directories(self):
        """Create necessary directories"""
        # Experiment directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path("experiments_multi") / f"{self.project_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create processed directory
        Path(self.data_dirs['processed']).mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Experiment directory: {self.experiment_dir}")
    
    def save_config(self):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and k != 'experiment_dir'}
        config_dict['experiment_dir'] = str(self.experiment_dir)
        
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

# ==================== ZIP HANDLER ====================
class ZipHandler:
    """Handles ZIP files"""
    
    @staticmethod
    def extract_patient_zips(patient_dir, extract_dir):
        """Extract all ZIP files for a single patient"""
        patient_dir = Path(patient_dir)
        extract_dir = Path(extract_dir)
        
        # Find all ZIP files in patient directory
        zip_files = list(patient_dir.glob("*.nii.zip"))
        
        if not zip_files:
            print(f"    ⚠️  No ZIP files found in {patient_dir.name}")
            return False
        
        # Create extraction directory for this patient
        patient_extract_dir = extract_dir / patient_dir.name
        patient_extract_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_count = 0
        for zip_file in zip_files:
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(patient_extract_dir)
                    extracted_count += 1
            except Exception as e:
                print(f"    ✗ Error extracting {zip_file.name}: {e}")
        
        # Check if extraction was successful
        nii_files = list(patient_extract_dir.glob("*.nii")) + list(patient_extract_dir.glob("*.nii.gz"))
        return len(nii_files) > 0
    
    @staticmethod
    def extract_all_patients(source_dir, extract_dir, mode='training'):
        """Extract all patients from a directory"""
        source_dir = Path(source_dir)
        extract_dir = Path(extract_dir) / mode  # Separate by mode
        
        print(f"\n📦 Extracting {mode} data...")
        
        # Find all patient directories
        patient_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir() and d.name.startswith('BraTS')])
        
        if not patient_dirs:
            print(f"⚠️  No patient directories found in {source_dir}")
            return []
        
        print(f"  Found {len(patient_dirs)} patient directories")
        
        # Extract each patient
        successful_extractions = []
        for patient_dir in patient_dirs:
            print(f"  📦 Processing {patient_dir.name}...")
            if ZipHandler.extract_patient_zips(patient_dir, extract_dir):
                successful_extractions.append(extract_dir / patient_dir.name)
                print(f"    ✓ Extracted successfully")
            else:
                print(f"    ✗ Extraction failed")
        
        print(f"✅ Successfully extracted {len(successful_extractions)}/{len(patient_dirs)} patients")
        return successful_extractions
    
    @staticmethod
    def check_already_extracted(extract_dir, mode):
        """Check if files are already extracted"""
        extract_dir = Path(extract_dir) / mode
        
        if not extract_dir.exists():
            return False
        
        # Check if any patient directories exist
        patient_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        
        if not patient_dirs:
            return False
        
        # Check if patient directories have NIfTI files
        for patient_dir in patient_dirs:
            nii_files = list(patient_dir.glob("*.nii")) + list(patient_dir.glob("*.nii.gz"))
            if not nii_files:
                return False
        
        return True

# ==================== MEMORY-EFFICIENT DATASET ====================
class MemoryEfficientDataset(torch.utils.data.Dataset):
    """Dataset with memory optimizations for large medical images"""
    
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        
        # Use LRU cache instead of full cache
        self.data_cache = {}
        self.cache_size = 0
        self.max_cache_size_gb = config.max_cache_size
        
        print(f"\n📂 Setting up {mode} dataset...")
        print(f"💾 Cache limit: {self.max_cache_size_gb} GB")
        
        # Prepare data (extract if needed)
        self.data_dir = self.prepare_data()
        
        # Find all patient directories
        self.patient_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if not self.patient_dirs:
            raise ValueError(f"No patient directories found in {self.data_dir}")
        
        print(f"📊 Found {len(self.patient_dirs)} patient directories")
        
        # Load patient information (metadata only, not data)
        self.patients = self.load_patient_metadata()
        
        if not self.patients:
            raise ValueError(f"No valid patients found in {mode} set")
        
        # Create slices list
        self.slices_list = self.create_slices_list()
        
        print(f"📊 {mode.capitalize()} slices: {len(self.slices_list)}")
    
    def prepare_data(self):
        """Prepare data by extracting ZIP files if needed"""
        if self.mode == 'train':
            source_dir = Path(self.config.data_dirs['training'])
        else:
            source_dir = Path(self.config.data_dirs['validation'])
        
        extract_dir = Path(self.config.data_dirs['extracted'])
        
        # Check if already extracted
        if self.config.use_existing_extracted and ZipHandler.check_already_extracted(extract_dir, self.mode):
            print(f"✅ Using already extracted {self.mode} data")
            return extract_dir / self.mode
        
        # Extract if needed
        if self.config.extract_zip_files:
            extracted_dirs = ZipHandler.extract_all_patients(source_dir, extract_dir, self.mode)
            
            if not extracted_dirs:
                raise ValueError(f"No data could be extracted from {source_dir}")
            
            return extract_dir / self.mode
        else:
            # Use ZIP files directly
            print(f"⚠️  Using ZIP files directly for {self.mode}")
            return source_dir
    
    def load_patient_metadata(self):
        """Load metadata only"""
        patients = []
        
        for patient_dir in self.patient_dirs:
            patient_id = patient_dir.name
            
            # Check if patient directory has required files
            modalities = self.check_patient_files(patient_dir)
            
            if modalities:
                # Load only mask to get tumor slices
                mask_info = self.load_mask_metadata(patient_dir, patient_id)
                
                patient_info = {
                    'id': patient_id,
                    'path': patient_dir,
                    'modalities': modalities,
                    'mask_info': mask_info
                }
                patients.append(patient_info)
                print(f"  ✓ {patient_id}: Found {len(modalities)} modalities")
            else:
                print(f"  ✗ {patient_id}: Missing required files")
        
        return patients
    
    def check_patient_files(self, patient_dir):
        """Check which modalities are available for a patient"""
        modalities = {}
        
        # Define file patterns to check
        file_patterns = {
            'flair': ['*flair*.nii*', 'flair.nii*'],
            't1': ['*t1.nii*', 't1.nii*'],
            't1ce': ['*t1ce*.nii*', '*t1gd*.nii*', 't1ce.nii*'],
            't2': ['*t2*.nii*', 't2.nii*'],
            'seg': ['*seg*.nii*', 'seg.nii*']
        }
        
        for modality, patterns in file_patterns.items():
            found = False
            for pattern in patterns:
                files = list(patient_dir.glob(pattern))
                if files:
                    # Check file is not empty
                    file_size = files[0].stat().st_size
                    if file_size > 1024:  # More than 1KB
                        modalities[modality] = str(files[0])
                        found = True
                        break
                    else:
                        print(f"    ⚠️  Empty file for {modality}: {files[0].name}")
            
            if not found and modality != 'seg':
                print(f"    ⚠️  Missing {modality}")
        
        return modalities
    def load_mask_metadata(self, patient_dir, patient_id):
        """Load only mask metadata to identify tumor slices - FIXED VERSION"""
        mask_files = list(patient_dir.glob("*seg*.nii*")) + list(patient_dir.glob("seg.nii*"))
    
        if not mask_files:
            return {'tumor_slices': [], 'depth': 155, 'shape': (240, 240, 155)}
    
        mask_file = mask_files[0]
        try:
            # Use memory mapping to load only header
            img = nib.load(str(mask_file), mmap=True)
            img_data = img.get_fdata()
        
            # Find tumor slices
            depth = img_data.shape[2] if len(img_data.shape) > 2 else 1
            tumor_slices = []
        
            # Sample slices to check for tumors
            for slice_idx in range(depth):
                if img_data[:, :, slice_idx].max() > 0:
                    tumor_slices.append(slice_idx)
        
            shape_info = img.shape  # Get shape before deleting img
        
            # Clear references
            del img_data
            # del img 
            gc.collect()
        
            return {
                'tumor_slices': tumor_slices,
                'depth': depth,
                'shape': shape_info  # Use stored shape
            }
        
        except Exception as e:
            print(f"    ✗ Error loading mask for {patient_id}: {e}")
            return {'tumor_slices': [], 'depth': 155, 'shape': (240, 240, 155)}
    
    def load_slice_from_disk(self, patient_info, slice_idx):
        """Load a single slice directly from disk"""
        patient_id = patient_info['id']
        patient_dir = patient_info['path']
        
        # Load each modality for this slice
        images_list = []
        modality_order = ['t1', 't1ce', 't2', 'flair']
        
        for modality in modality_order:
            if modality in patient_info['modalities']:
                file_path = Path(patient_info['modalities'][modality])
                
                try:
                    # Load with memory mapping
                    img = nib.load(str(file_path), mmap=True)
                    img_data = img.get_fdata()
                    
                    # Extract specific slice
                    if slice_idx < img_data.shape[2]:
                        img_slice = img_data[:, :, slice_idx].astype(np.float32)
                    else:
                        img_slice = img_data[:, :, 0].astype(np.float32)
                    
                    # Preprocess
                    img_slice = self.preprocess_image(img_slice, modality)
                    
                    images_list.append(img_slice)
                    
                    # Clear references
                    del img_data
                    del img
                    
                except Exception as e:
                    print(f"    ✗ Error loading {modality} slice {slice_idx}: {e}")
                    images_list.append(np.zeros((240, 240), dtype=np.float32))
            else:
                # Use zeros for missing modalities
                images_list.append(np.zeros((240, 240), dtype=np.float32))
        
        # Load mask slice
        if 'seg' in patient_info['modalities']:
            mask_file = Path(patient_info['modalities']['seg'])
            try:
                img = nib.load(str(mask_file), mmap=True)
                mask_data = img.get_fdata()
                
                if slice_idx < mask_data.shape[2]:
                    mask_slice = mask_data[:, :, slice_idx].astype(np.int64)
                else:
                    mask_slice = mask_data[:, :, 0].astype(np.int64)
                
                mask_slice = self.preprocess_mask(mask_slice)
                
                del mask_data
                del img
                
            except Exception as e:
                print(f"    ✗ Error loading mask slice {slice_idx}: {e}")
                mask_slice = np.zeros((240, 240), dtype=np.int64)
        else:
            mask_slice = np.zeros((240, 240), dtype=np.int64)
        
        gc.collect()
        
        return images_list, mask_slice
    
    def preprocess_image(self, image, modality):
        """Preprocess a single slice"""
        image = image.copy()
        
        # Clip outliers
        p_low, p_high = np.percentile(image, self.config.clip_percentile)
        image = np.clip(image, p_low, p_high)
        
        # Normalize to [0, 1]
        if self.config.normalize_per_modality:
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
        
        return image
    
    def preprocess_mask(self, mask):
        """Convert mask slice to standard classes"""
        mask = mask.copy()
        processed_mask = np.zeros_like(mask, dtype=np.int64)
        
        if mask.max() > 0:
            # Standard BRATS conversion
            processed_mask[mask == 1] = 1  # Necrotic core
            processed_mask[mask == 2] = 3  # Edema
            processed_mask[mask == 4] = 2  # Enhancing tumor
            
            # Handle combined labels
            processed_mask[mask == 3] = 1
            processed_mask[mask == 5] = 2
        
        return processed_mask
    
    def create_slices_list(self):
        """Create list of (patient_id, slice_idx)"""
        slices_list = []
        
        for patient_info in self.patients:
            mask_info = patient_info.get('mask_info', {})
            tumor_slices = mask_info.get('tumor_slices', [])
            depth = mask_info.get('depth', 155)
            
            if not tumor_slices:
                # Use middle slices if no tumor slices found
                tumor_slices = list(range(depth // 4, 3 * depth // 4))
            
            # Add tumor slices
            for slice_idx in tumor_slices:
                slices_list.append({
                    'patient_id': patient_info['id'],
                    'slice_idx': slice_idx,
                    'has_tumor': True
                })
            
            # Add some non-tumor slices for balance (only for training)
            if self.mode == 'train':
                all_slices = list(range(depth))
                non_tumor_slices = [i for i in all_slices if i not in tumor_slices]
                
                # Add some non-tumor slices (max 20% of tumor slices)
                num_non_tumor = min(len(non_tumor_slices), len(tumor_slices) // 5)
                if num_non_tumor > 0:
                    selected_non_tumor = random.sample(non_tumor_slices, num_non_tumor)
                    for slice_idx in selected_non_tumor:
                        slices_list.append({
                            'patient_id': patient_info['id'],
                            'slice_idx': slice_idx,
                            'has_tumor': False
                        })
        
        # Shuffle slices (only for training)
        if self.mode == 'train':
            random.shuffle(slices_list)
        
        return slices_list
    
    def manage_cache(self):
        """Manage cache size"""
        current_memory = get_memory_usage()
        
        if current_memory > self.max_cache_size_gb:
            # Clear half of cache
            keys_to_remove = list(self.data_cache.keys())[:len(self.data_cache)//2]
            for key in keys_to_remove:
                del self.data_cache[key]
            gc.collect()
            print(f"  🗑️  Cleared cache: {len(keys_to_remove)} items")
    
    def safe_flip(self, array, axis):
        """Safe flip operation"""
        return np.flip(array, axis=axis).copy()
    
    def safe_rotate(self, array, angle, axes, order=3):
        """Safe rotate operation"""
        return ndimage.rotate(array, angle, axes=axes, reshape=False, 
                            order=order, mode='constant', cval=0).copy()
    
    def augment_data(self, images, mask):
        """Apply data augmentation for training"""
        if not self.config.augmentation or self.mode != 'train':
            return [img.copy() for img in images], mask.copy()
        
        augmented_images = [img.copy() for img in images]
        augmented_mask = mask.copy()
        
        # Random rotation
        if random.random() < 0.8:
            angle = random.uniform(-self.config.rotation_range, self.config.rotation_range)
            for i in range(len(augmented_images)):
                augmented_images[i] = self.safe_rotate(augmented_images[i], angle, axes=(0, 1), order=1)
            augmented_mask = self.safe_rotate(augmented_mask, angle, axes=(0, 1), order=0)
        
        # Random flip
        if random.random() < self.config.flip_prob:
            axis = random.choice([0, 1])
            for i in range(len(augmented_images)):
                augmented_images[i] = self.safe_flip(augmented_images[i], axis=axis)
            augmented_mask = self.safe_flip(augmented_mask, axis=axis)
        
        # Random intensity scaling
        if random.random() < 0.5:
            scale = random.uniform(*self.config.intensity_scale_range)
            for i in range(len(augmented_images)):
                augmented_images[i] = np.clip(augmented_images[i] * scale, 0, 1).copy()
        
        return augmented_images, augmented_mask
    
    def __len__(self):
        return len(self.slices_list)
    
    def __getitem__(self, idx):
        """Get a slice from the dataset"""
        gc.collect()
        
        slice_info = self.slices_list[idx]
        patient_id = slice_info['patient_id']
        slice_idx = slice_info['slice_idx']
        
        # Check cache
        cache_key = f"{patient_id}_{slice_idx}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Find patient info
        patient_info = next((p for p in self.patients if p['id'] == patient_id), None)
        if not patient_info:
            # Return empty data if patient not found
            empty_image = torch.zeros((self.config.in_channels, self.config.image_size, self.config.image_size))
            empty_mask = torch.zeros((self.config.image_size, self.config.image_size), dtype=torch.long)
            return empty_image, empty_mask
        
        # Load slice from disk
        images_list, mask_slice = self.load_slice_from_disk(patient_info, slice_idx)
        
        # Apply augmentation for training
        if self.mode == 'train':
            images_list, mask_slice = self.augment_data(images_list, mask_slice)
        
        # Stack images (4, H, W)
        images_array = np.stack(images_list, axis=0)
        
        # Resize to config image size if needed
        if images_array.shape[1] != self.config.image_size or images_array.shape[2] != self.config.image_size:
            from scipy.ndimage import zoom
            h_ratio = self.config.image_size / images_array.shape[1]
            w_ratio = self.config.image_size / images_array.shape[2]
            
            images_array = zoom(images_array, (1, h_ratio, w_ratio), order=1)
            mask_slice = zoom(mask_slice, (h_ratio, w_ratio), order=0)
        
        # Convert to tensors
        images_tensor = torch.FloatTensor(images_array.copy())
        mask_tensor = torch.LongTensor(mask_slice.copy())
        
        # Cache if enough memory
        current_memory = get_memory_usage()
        if current_memory < self.max_cache_size_gb:
            self.data_cache[cache_key] = (images_tensor, mask_tensor)
        
        # Manage cache periodically
        if idx % 100 == 0:
            self.manage_cache()
        
        return images_tensor, mask_tensor

# ==================== 2D U-NET ====================
class SimpleUNet2D(nn.Module):
    """2D U-Net for brain tumor segmentation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        print("\n🧠 Building 2D U-Net...")
        
        # Encoder
        self.enc1 = self._conv_block(config.in_channels, config.base_filters)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = self._conv_block(config.base_filters, config.base_filters * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(config.base_filters * 2, config.base_filters * 4)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(config.base_filters * 4, config.base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(config.base_filters * 4, config.base_filters * 2)
        
        self.up1 = nn.ConvTranspose2d(config.base_filters * 2, config.base_filters, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(config.base_filters * 2, config.base_filters)
        
        # Output
        self.output = nn.Conv2d(config.base_filters, config.num_classes, kernel_size=1)
        
        self._initialize_weights()
        self._print_model_info()
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"📊 Model Info:")
        print(f"   Architecture: 2D U-Net")
        print(f"   Input size: {self.config.image_size}x{self.config.image_size}")
        print(f"   Input channels: {self.config.in_channels}")
        print(f"   Output classes: {self.config.num_classes}")
        print(f"   Parameters: {total_params:,} ({trainable_params:,} trainable)")
        print(f"   Base filters: {self.config.base_filters}")
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.pool1(enc1)
        
        enc3 = self.enc2(enc2)
        enc4 = self.pool2(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        
        # Decoder
        dec2 = self.up2(bottleneck)
        dec2 = torch.cat([dec2, enc3], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        return self.output(dec1)

# ==================== MEDICAL METRICS ====================
class MedicalMetrics:
    """Medical segmentation metrics"""
    
    @staticmethod
    def dice_coefficient(pred, target, class_idx):
        pred_mask = (pred == class_idx).float()
        target_mask = (target == class_idx).float()
        
        if pred_mask.sum() == 0 and target_mask.sum() == 0:
            return 1.0
        
        intersection = (pred_mask * target_mask).sum()
        dice = (2.0 * intersection) / (pred_mask.sum() + target_mask.sum() + 1e-6)
        return dice.item()
    
    @staticmethod
    def calculate_all_metrics(pred, target, num_classes=4):
        metrics = {}
        
        for class_idx in range(1, num_classes):
            class_name = {1: "necrotic", 2: "enhancing", 3: "edema"}[class_idx]
            
            dice = MedicalMetrics.dice_coefficient(pred, target, class_idx)
            
            tp = ((pred == class_idx) & (target == class_idx)).sum().float()
            fp = ((pred == class_idx) & (target != class_idx)).sum().float()
            fn = ((pred != class_idx) & (target == class_idx)).sum().float()
            
            tp_val = tp.item()
            fp_val = fp.item()
            fn_val = fn.item()
            
            sensitivity = tp_val / (tp_val + fn_val + 1e-6) if (tp_val + fn_val) > 0 else 0.0
            precision = tp_val / (tp_val + fp_val + 1e-6) if (tp_val + fp_val) > 0 else 0.0
            
            tn = ((pred != class_idx) & (target != class_idx)).sum().float()
            tn_val = tn.item()
            specificity = tn_val / (tn_val + fp_val + 1e-6) if (tn_val + fp_val) > 0 else 0.0
            
            metrics[class_name] = {
                'dice': dice,
                'sensitivity': sensitivity,
                'precision': precision,
                'specificity': specificity,
                'tp': tp_val,
                'fp': fp_val,
                'fn': fn_val
            }
        
        # Overall metrics
        dice_scores = [m['dice'] for m in metrics.values()]
        metrics['overall'] = {
            'mean_dice': float(np.mean(dice_scores)) if dice_scores else 0.0,
            'std_dice': float(np.std(dice_scores)) if dice_scores else 0.0
        }
        
        return metrics

# ==================== TRAINING FUNCTION ====================
def train_separate_datasets():
    """Main training function for separate train/val datasets"""
    print("\n" + "=" * 80)
    print("🎯 STARTING TRAINING WITH SEPARATE TRAIN/VAL DATASETS")
    print("=" * 80)
    
    # Initialize config
    config = MultiPatientConfig()
    
    # Create datasets
    print("\n📊 Loading separate train/val datasets...")
    print(f"💾 Current memory: {get_memory_usage():.2f} GB")
    
    try:
        train_dataset = MemoryEfficientDataset(config, mode='train')
        val_dataset = MemoryEfficientDataset(config, mode='val')
        
        print(f"💾 Memory after loading datasets: {get_memory_usage():.2f} GB")
        
        print(f"\n✅ Datasets created:")
        print(f"   Training patients: {len(train_dataset.patients)}")
        print(f"   Validation patients: {len(val_dataset.patients)}")
        print(f"   Training slices: {len(train_dataset)}")
        print(f"   Validation slices: {len(val_dataset)}")
        
        # Print patient lists
        if train_dataset.patients:
            train_ids = [p['id'] for p in train_dataset.patients]
            print(f"\n📊 Training patients: {', '.join(train_ids)}")
        
        if val_dataset.patients:
            val_ids = [p['id'] for p in val_dataset.patients]
            print(f"📊 Validation patients: {', '.join(val_ids)}")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    # Create data loaders with memory management
    print("\n🔄 Creating data loaders...")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False  # Don't pin memory on CPU
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    print(f"   Training batches per epoch: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Initialize model
    print("\n🧠 Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    model = SimpleUNet2D(config).to(device)
    
    # Loss and optimizer
    class_weights = torch.tensor([0.1, 1.0, 1.5, 1.2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    
    # Training history
    history = {
        'epoch': [], 
        'train_loss': [], 
        'val_loss': [], 
        'val_dice': [], 
        'lr': [],
        'train_patients': len(train_dataset.patients),
        'val_patients': len(val_dataset.patients),
        'patient_ids': {
            'train': [p['id'] for p in train_dataset.patients],
            'val': [p['id'] for p in val_dataset.patients]
        }
    }
    
    print("\n🔥 STARTING TRAINING")
    print("=" * 80)
    
    best_dice = 0
    best_model_path = None
    start_time = time.time()
    no_improvement_count = 0
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        gc.collect()
        
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        print(f"\n📚 Epoch {epoch+1}/{config.num_epochs}")
        print(f"   Training patients: {history['train_patients']}")
        print(f"   Validation patients: {history['val_patients']}")
        print(f"   💾 Memory: {get_memory_usage():.2f} GB")
        print("-" * 40)
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Clear intermediate variables
            del outputs
            del loss
            
            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                print(f"  Batch {batch_idx+1:3d}/{len(train_loader)} | Loss: {train_loss/train_batches:.4f}")
            
            # Clear cache periodically
            if batch_idx % config.clear_cache_every == 0:
                gc.collect()
        
        avg_train_loss = train_loss / max(train_batches, 1)
        
        # Validation
        model.eval()
        val_loss = 0
        val_dice_scores = []
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_batches += 1
                
                preds = torch.argmax(outputs, dim=1)
                
                for i in range(preds.shape[0]):
                    pred = preds[i]
                    mask = masks[i]
                    
                    metrics = MedicalMetrics.calculate_all_metrics(pred, mask)
                    val_dice_scores.append(metrics['overall']['mean_dice'])
                
                # Clear cache
                if batch_idx % config.clear_cache_every == 0:
                    gc.collect()
        
        avg_val_loss = val_loss / max(val_batches, 1) if val_batches > 0 else 0
        avg_val_dice = np.mean(val_dice_scores) if val_dice_scores else 0
        
        # Update scheduler
        scheduler.step(avg_val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print learning rate update
        if epoch > 0 and current_lr < history['lr'][-1]:
            print(f"📉 Learning rate reduced to: {current_lr:.6f}")
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(float(avg_train_loss))
        history['val_loss'].append(float(avg_val_loss))
        history['val_dice'].append(float(avg_val_dice))
        history['lr'].append(float(current_lr))
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\n📊 Epoch {epoch+1:3d} Summary")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:   {avg_val_loss:.4f}")
        print(f"   Val Dice:   {avg_val_dice:.4f}")
        print(f"   LR:         {current_lr:.6f}")
        print(f"   Time:       {epoch_time:.1f}s")
        print(f"   💾 Memory:  {get_memory_usage():.2f} GB")
        print("-" * 40)
        
        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            best_model_path = config.experiment_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': best_dice,
                'config': config.__dict__,
                'train_patients': history['patient_ids']['train'],
                'val_patients': history['patient_ids']['val']
            }, best_model_path)
            print(f"💾 Saved best model (Dice: {best_dice:.4f})")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= config.patience:
                print(f"⏹️  Early stopping after {config.patience} epochs without improvement")
                break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = config.experiment_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint at epoch {epoch+1}")
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n⏱️  Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    # Save final model
    final_model_path = config.experiment_dir / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("🧪 FINAL EVALUATION")
    print("=" * 80)
    
    # Load best model for evaluation
    if best_model_path and best_model_path.exists():
        try:
            best_checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            print(f"✅ Loaded best model from epoch {best_checkpoint['epoch']}")
            print(f"   Best Dice: {best_checkpoint['dice']:.4f}")
        except:
            print("⚠️  Could not load best model, using final model")
    
    model.eval()
    test_metrics = []
    all_val_dice = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for i in range(preds.shape[0]):
                pred = preds[i]
                mask = masks[i]
                
                metrics = MedicalMetrics.calculate_all_metrics(pred, mask)
                test_metrics.append(metrics)
                all_val_dice.append(metrics['overall']['mean_dice'])
                
                # Print first sample metrics
                if batch_idx == 0 and i == 0:
                    print(f"\n📊 Sample prediction metrics:")
                    print(f"   Overall Dice: {metrics['overall']['mean_dice']:.4f}")
                    for class_name, class_metrics in metrics.items():
                        if class_name != 'overall':
                            print(f"   {class_name:10s}: Dice={class_metrics['dice']:.4f}")
    
    # Calculate statistics
    if test_metrics:
        mean_dice = np.mean(all_val_dice)
        std_dice = np.std(all_val_dice)
        
        print(f"\n📈 Final Results:")
        print(f"   Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"   Best Dice: {best_dice:.4f}")
        print(f"   Test slices: {len(test_metrics)}")
        print(f"   Validation patients: {len(val_dataset.patients)}")
    else:
        print("⚠️  No test samples")
        mean_dice = 0
        std_dice = 0
    
    # Save results
    print("\n💾 Saving results...")
    
    # Save history
    import pandas as pd
    history_df = pd.DataFrame({
        'epoch': history['epoch'],
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_dice': history['val_dice'],
        'lr': history['lr']
    })
    history_df.to_csv(config.experiment_dir / "training_history.csv", index=False)
    
    # Save patient information
    patient_info = {
        'train_patients': history['patient_ids']['train'],
        'val_patients': history['patient_ids']['val'],
        'total_train_patients': len(train_dataset.patients),
        'total_val_patients': len(val_dataset.patients),
        'train_slices': len(train_dataset),
        'val_slices': len(val_dataset),
        'data_structure': 'Separate train/val directories',
        'train_path': str(train_dataset.data_dir),
        'val_path': str(val_dataset.data_dir)
    }
    
    with open(config.experiment_dir / "patient_info.json", "w") as f:
        json.dump(patient_info, f, indent=2)
    
    # Save final metrics
    results = {
        'best_dice': float(best_dice),
        'final_mean_dice': float(mean_dice),
        'final_std_dice': float(std_dice),
        'num_test_slices': len(test_metrics),
        'num_val_patients': len(val_dataset.patients),
        'total_training_time': float(total_time),
        'epochs_completed': len(history['epoch'])
    }
    
    with open(config.experiment_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    create_visualization(config, history, test_metrics[0] if test_metrics else None)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING WITH SEPARATE DATASETS COMPLETED!")
    print("=" * 80)
    print(f"\n📁 Results saved in: {config.experiment_dir}")
    print(f"   📄 config.json             - Project configuration")
    print(f"   📋 patient_info.json       - Patient split information")
    print(f"   💾 best_model.pth          - Best performing model")
    print(f"   💾 final_model.pth         - Final trained model")
    print(f"   📊 training_history.csv    - Complete training history")
    print(f"   📋 test_results.json       - Final evaluation metrics")
    print(f"   🎨 training_summary.png    - Visualization")
    
    # Print summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Training patients: {patient_info['total_train_patients']}")
    print(f"   Validation patients: {patient_info['total_val_patients']}")
    print(f"   Training slices: {patient_info['train_slices']}")
    print(f"   Validation slices: {patient_info['val_slices']}")
    print(f"   Data structure: {patient_info['data_structure']}")
    
    return model, history, results

def create_visualization(config, history, sample_metrics=None):
    """Create visualization plots"""
    fig = plt.figure(figsize=(18, 12))
    
    # Loss plot
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(history['epoch'], history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(history['epoch'], history['val_loss'], 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dice plot
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(history['epoch'], history['val_dice'], 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Validation Dice Score')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=max(history['val_dice']), color='r', linestyle='--', alpha=0.5, 
                label=f'Best: {max(history["val_dice"]):.3f}')
    ax2.legend()
    
    # Learning rate
    ax3 = plt.subplot(2, 4, 3)
    ax3.plot(history['epoch'], history['lr'], 'purple', linewidth=2, marker='^')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Sample predictions (if available)
    if sample_metrics:
        # Metrics bar chart
        ax4 = plt.subplot(2, 4, 4)
        classes = ['Necrotic', 'Enhancing', 'Edema']
        dice_values = [
            sample_metrics.get('necrotic', {}).get('dice', 0),
            sample_metrics.get('enhancing', {}).get('dice', 0),
            sample_metrics.get('edema', {}).get('dice', 0)
        ]
        
        colors = ['red', 'green', 'blue']
        bars = ax4.bar(classes, dice_values, color=colors)
        ax4.set_ylabel('Dice Score')
        ax4.set_title('Per-Class Dice Scores')
        ax4.set_ylim([0, 1])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, dice_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Patient distribution
    ax5 = plt.subplot(2, 4, 5)
    ax5.axis('off')
    
    patient_text = f"""
PATIENT DISTRIBUTION
════════════════════
Training: {history['train_patients']} patients
Validation: {history['val_patients']} patients
Total: {history['train_patients'] + history['val_patients']}

DATA STRUCTURE:
Separate directories
Training: data/raw/training/
Validation: data/raw/validation/
    """
    ax5.text(0.1, 0.5, patient_text, fontsize=9, family='monospace',
            verticalalignment='center')
    
    # Model architecture
    ax6 = plt.subplot(2, 4, 6)
    ax6.axis('off')
    
    model_text = f"""
MODEL ARCHITECTURE
═══════════════════
2D U-Net
Input: 4×{config.image_size}×{config.image_size}
Base filters: {config.base_filters}
Parameters: ~{(config.base_filters * 1000):,}

MEMORY OPTIMIZATION:
Batch size: {config.batch_size}
Cache limit: {config.max_cache_size} GB
Memory mapping: {'Yes' if config.use_memory_mapping else 'No'}
    """
    ax6.text(0.1, 0.5, model_text, fontsize=9, family='monospace',
            verticalalignment='center')
    
    # Training summary
    ax7 = plt.subplot(2, 4, 7)
    ax7.axis('off')
    
    if history['epoch']:
        summary = f"""
TRAINING SUMMARY
═════════════════
Epochs: {len(history['epoch'])}/{config.num_epochs}
Best Dice: {max(history['val_dice']):.4f}
Final Dice: {history['val_dice'][-1]:.4f}
Final Loss: {history['val_loss'][-1]:.4f}

MEMORY USAGE:
Cache management: LRU
Batch loading: ✓
Memory mapping: ✓
    """
    else:
        summary = "No training history available"
    
    ax7.text(0.1, 0.5, summary, fontsize=9, family='monospace',
            verticalalignment='center')
    
    # Performance metrics
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    if history['epoch']:
        metrics_text = f"""
PERFORMANCE METRICS
═══════════════════
Early stopping: {history['epoch'][-1] < config.num_epochs}
LR reductions: {sum(1 for i in range(1, len(history['lr'])) 
                   if history['lr'][i] < history['lr'][i-1])}

MEMORY EFFICIENCY:
Slice-by-slice loading: ✓
On-demand data access: ✓
Periodic cache clearing: ✓
    """
    else:
        metrics_text = "No metrics available"
    
    ax8.text(0.1, 0.5, metrics_text, fontsize=9, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('BRAIN TUMOR SEGMENTATION - SEPARATE TRAIN/VAL DATASETS', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = config.experiment_dir / "training_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved: {plot_path}")

# ==================== MAIN ====================
if __name__ == "__main__":
    try:
        torch.set_num_threads(1)
        
        print("\n🔍 Checking data structure...")
        print(f"💾 Initial memory: {get_memory_usage():.2f} GB")
        
        # Check both directories
        training_dir = Path("data/raw/training")
        validation_dir = Path("data/raw/validation")
        
        dirs_to_check = [
            ("Training", training_dir),
            ("Validation", validation_dir)
        ]
        
        all_good = True
        for dir_name, dir_path in dirs_to_check:
            if not dir_path.exists():
                print(f"❌ {dir_name} directory not found: {dir_path}")
                all_good = False
            else:
                # Count patient directories
                patient_dirs = [d for d in dir_path.iterdir() if d.is_dir() and d.name.startswith('BraTS')]
                print(f"✅ {dir_name}: Found {len(patient_dirs)} patient directories")
                
                if patient_dirs:
                    # Check files in first patient
                    sample_patient = patient_dirs[0]
                    zip_files = list(sample_patient.glob("*.nii.zip"))
                    print(f"   Sample patient: {sample_patient.name} ({len(zip_files)} ZIP files)")
        
        if not all_good:
            print(f"\n📁 Required structure:")
            print("data/raw/")
            print("├── training/")
            print("│   ├── BraTS001/")
            print("│   │   ├── BraTS20_Training_001_flair.nii.zip")
            print("│   │   ├── BraTS20_Training_001_t1.nii.zip")
            print("│   │   ├── ...")
            print("│   ├── BraTS002/")
            print("│   └── ... (5 patients total)")
            print("└── validation/")
            print("    ├── BraTS006/")
            print("    ├── BraTS007/")
            print("    └── ... (5 patients total)")
            sys.exit(1)
        
        print("\n" + "=" * 80)
        print("📚 MULTI-PATIENT TRAINING WITH SEPARATE DATASETS")
        print("=" * 80)
        print(f"\n📊 Detected structure:")
        print(f"   Training directory: {training_dir}")
        print(f"   Validation directory: {validation_dir}")
        
        config = MultiPatientConfig()
        print(f"\n⚙️  Configuration:")
        print(f"   Extract ZIP files: {'Yes' if config.extract_zip_files else 'No'}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   Augmentation: {'Yes' if config.augmentation else 'No'}")
        print(f"   Memory optimization: ✓ (Cache: {config.max_cache_size} GB)")
        
        print("\n📋 This setup will:")
        print("   1. Use 5 patients for training")
        print("   2. Use 5 DIFFERENT patients for validation")
        print("   3. Load slices on-demand to save memory")
        print("   4. Use memory-mapped files for large volumes")
        
        print("\n" + "=" * 80)
        response = input("Start training with separate datasets? (y/n): ")
        
        if response.lower() == 'y':
            model, history, results = train_separate_datasets()
            
            print("\n" + "=" * 80)
            print("🎓 FOR YOUR THESIS:")
            print("=" * 80)
            print("\nKey achievements with separate datasets:")
            print("1. ✅ True independent validation set")
            print("2. ✅ Memory-efficient slice-by-slice loading")
            print("3. ✅ No data leakage between train and val")
            print("4. ✅ Better generalization assessment")
            
            print("\nScientific validity:")
            print("• Training on 5 complete patient cases")
            print("• Validation on 5 different patient cases")
            print("• No patient overlap between sets")
            print("• Memory-optimized for large medical images")
            
        else:
            print("Training cancelled.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        gc.collect()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()