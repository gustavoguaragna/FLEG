import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, ConcatDataset
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from flwr_datasets.partitioner.partitioner import Partitioner
from typing import Mapping, Iterable, Any, Literal, Union, List, Tuple
from typing import Optional, List, Dict, Tuple
from collections import defaultdict, Counter
from datasets import Dataset
import numpy as np
import random
import math

class Net(nn.Module):
    """
    Rede Neural Convolucional base para o dataset MNIST.
    Composta por duas camadas convolucionais seguidas por três camadas lineares (fully connected layers).
    """
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor1(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(FeatureExtractor1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        return x
    
class ClassifierHead1(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(ClassifierHead1, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84) 
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        if x.dim() == 2:
              x = x.view(-1, 6, 12, 12)

        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor2(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(FeatureExtractor2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        return x

class ClassifierHead2(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(ClassifierHead2, self).__init__()
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor3(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(FeatureExtractor3, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4) 
        x = F.relu(self.fc1(x))
        return x

class ClassifierHead3(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(ClassifierHead3, self).__init__()
        self.fc2 = nn.Linear(120, 84) 
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor4(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(FeatureExtractor4, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ClassifierHead4(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(ClassifierHead4, self).__init__()
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        x = self.fc3(x)
        return x
    
class Net_Cifar(nn.Module):
    """
    Rede Neural Convolucional base para o dataset CIFAR-10.
    Adaptada para lidar com imagens coloridas (3 canais).
    """
    def __init__(self,seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor1_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(FeatureExtractor1_Cifar, self).__init__()
        # Input: 3 canais (RGB)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        return x

class ClassifierHead1_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(ClassifierHead1_Cifar, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 6, 14, 14)

        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor2_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(FeatureExtractor2_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return x

class ClassifierHead2_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(ClassifierHead2_Cifar, self).__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor3_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(FeatureExtractor3_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

class ClassifierHead3_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(ClassifierHead3_Cifar, self).__init__()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor4_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(FeatureExtractor4_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ClassifierHead4_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(ClassifierHead4_Cifar, self).__init__()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.fc3(x)
        return x

class BaseEmbeddingGAN(nn.Module):
    """
    Classe base para a GAN
    """
    def __init__(self):
        super(BaseEmbeddingGAN, self).__init__()
        self.adv_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_tensor, labels=None):
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.view(input_tensor.size(0), -1)

        if input_tensor.shape[1] == self.latent_dim:
            if self.condition and labels is not None:
                embedded_labels = self.label_embedding(labels)
                gen_input = torch.cat((input_tensor, embedded_labels), dim=1)
                return self.generator(gen_input)
            else:
                return self.generator(input_tensor)

        elif input_tensor.shape[1] == self.embedding_dim:
            if self.condition and labels is not None:
                embedded_labels = self.label_embedding(labels)
                disc_input = torch.cat((input_tensor, embedded_labels), dim=1)
                return self.discriminator(disc_input)
            else:
                return self.discriminator(input_tensor)
        else:
            raise ValueError(f"Input tensor shape {input_tensor.shape} invalid. Expected dim {self.latent_dim} (Gen) or {self.embedding_dim} (Disc).")

    def loss(self, output, label):
        return self.adv_loss(output, label)

# --- GAN NÍVEL 1 (Embedding Dim: 864) ---
class EmbeddingGAN1(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=864, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN1, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

# --- GAN NÍVEL 2 (Embedding Dim: 256) ---
class EmbeddingGAN2(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=256, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN2, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

# --- GAN NÍVEL 3 (Embedding Dim: 120) ---
class EmbeddingGAN3(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=120, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN3, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

# --- GAN NÍVEL 4 (Embedding Dim: 84) ---
class EmbeddingGAN4(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=84, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN4, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )

# --- GAN NÍVEL 1 CIFAR (Embedding Dim: 1176) ---
class EmbeddingGAN1_Cifar(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=1176, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN1_Cifar, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

class EmbeddingGAN2_Cifar(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=400, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN2_Cifar, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

class EmbeddingGAN3_Cifar(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=120, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN3_Cifar, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )
    
class EmbeddingGAN4_Cifar(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=84, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN4_Cifar, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )

class GeneratedAssetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 generator,
                 num_samples,
                 latent_dim=128,
                 num_classes=10,
                 asset_shape=None,
                 desired_classes=None,
                 device=torch.device("cpu"),
                 asset_col_name="image",
                 label_col_name="label"):
        """
        Gera um dataset de ativos (imagens ou embeddings) usando um modelo gerativo condicional.

        Args:
            generator: O modelo gerativo pré-treinado.
            num_samples (int): Número total de ativos a serem gerados.
            latent_dim (int): Dimensão do vetor do espaço latente (z).
            num_classes (int): Número total de classes possíveis (usado para geração de rótulos).
            asset_shape (tuple, optional): Forma dos ativos gerados (ex: (1, 28, 28) para MNIST). Necessário se os ativos forem imagens.
            desired_classes (list[int], optional): Lista de índices de classes a serem geradas. Padrão é todas.
            device (torch.device): Dispositivo para executar a geração.
            asset_col_name (str): Nome para a coluna de ativo gerado.
            label_col_name (str): Nome para a coluna de rótulo.
        """
        self.generator = generator
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.total_num_classes = num_classes
        self.asset_shape = asset_shape
        self.device = device
        self.asset_col_name = asset_col_name
        self.label_col_name = label_col_name

        if desired_classes is not None and len(desired_classes) > 0:
            if not all(0 <= c < self.total_num_classes for c in desired_classes):
                raise ValueError(f"All desired classes must be integers between 0 and {self.total_num_classes - 1}")
            self._actual_classes_to_generate = sorted(list(set(desired_classes)))
        else:
            self._actual_classes_to_generate = list(range(self.total_num_classes))

        self.classes = self._actual_classes_to_generate
        self.num_generated_classes = len(self.classes)

        if self.num_generated_classes == 0 and self.num_samples > 0:
             raise ValueError("Cannot generate samples with an empty list of desired classes.")
        elif self.num_samples == 0:
             print("Warning: num_samples is 0. Dataset will be empty.")
             self.assets = torch.empty(0, *self.asset_shape) if self.asset_shape else torch.empty(0)
             self.labels = torch.empty(0, dtype=torch.long)
        else:
             if self.asset_shape is None:
                 raise ValueError("asset_shape must be provided when num_samples > 0.")
             self.assets, self.labels = self.generate_data()


    def generate_data(self):
        """Gera os dados do dataset."""
        self.generator.eval()
        self.generator.to(self.device)

        # Label generation logic
        generated_labels_list = []
        if self.num_generated_classes > 0:
            samples_per_class = self.num_samples // self.num_generated_classes
            remainder = self.num_samples % self.num_generated_classes
            for cls in self._actual_classes_to_generate:
                generated_labels_list.extend([cls] * samples_per_class)
            if remainder > 0:
                generated_labels_list.extend(random.choices(self._actual_classes_to_generate, k=remainder))
            random.shuffle(generated_labels_list)
        labels = torch.tensor(generated_labels_list, dtype=torch.long, device=self.device)

        z = torch.randn(self.num_samples, self.latent_dim, device=self.device)
        generated_assets_list = []
        batch_size = min(1024, self.num_samples) if self.num_samples > 0 else 1

        with torch.no_grad():
            for i in range(0, self.num_samples, batch_size):
                z_batch = z[i : i + batch_size]
                labels_batch = labels[i : i + batch_size]
                if z_batch.shape[0] == 0: continue

                gen_assets = self.generator(z_batch, labels_batch)
                generated_assets_list.append(gen_assets)

        if generated_assets_list:
            all_gen_assets = torch.cat(generated_assets_list, dim=0)
        else:
            print("Warning: No images generated. Returning empty tensor for images.")
            all_gen_assets = torch.empty(0, *self.asset_shape, device=self.device)

        return all_gen_assets, labels

    def __len__(self):
        return self.assets.shape[0]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Dataset index out of range")
        return {
            self.asset_col_name: self.assets[idx],
            self.label_col_name: int(self.labels[idx])
        }   

class ClassPartitioner(Partitioner):
    """Particiona um dataset em `num_partitions` partições baseadas em classes, garantindo que cada partição contenha classes exclusivas.

    Args:
        num_partitions (int): Número de partições a serem criadas (deve ser ≤ número de classes no dataset)
        seed (int, optional): Semente para embaralhamento aleatório das classes antes da divisão (padrão: None, sem embaralhamento)
        label_column (str): Nome da coluna contendo os rótulos das classes
    """

    def __init__(
        self,
        num_partitions: int,
        seed: Optional[int] = None,
        label_column: str = "label"
    ) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self._seed = seed
        self._label_column = label_column
        self._partition_indices: Optional[List[List[int]]] = None

    def _create_partitions(self) -> None:
        """Cria as partições do dataset com base nas classes, garantindo exclusividade de classes entre partições."""
        labels = self.dataset[self._label_column]

        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)

        classes = list(class_indices.keys())
        num_classes = len(classes)

        if self._num_partitions > num_classes:
            raise ValueError(
                f"Cannot create {self._num_partitions} partitions with only {num_classes} classes. "
                f"Reduce partitions to ≤ {num_classes}."
            )

        rng = random.Random(self._seed)
        rng.shuffle(classes)

        partition_classes = np.array_split(classes, self._num_partitions)

        self._partition_indices = []
        for class_group in partition_classes:
            indices = []
            for cls in class_group:
                indices.extend(class_indices[cls])
            self._partition_indices.append(indices)

    @property
    def dataset(self) -> Dataset:
        return super().dataset

    @dataset.setter
    def dataset(self, value: Dataset) -> None:
        super(ClassPartitioner, ClassPartitioner).dataset.fset(self, value)

        self._create_partitions()

    def load_partition(self, partition_id: int) -> Dataset:
        """Carrega uma partição contendo classes exclusivas.

        Args:
            partition_id: ID da partição a ser carregada (0-indexado)

        Retorna subconjunto do dataset original contendo apenas os índices correspondentes à partição especificada.
        """
        if not self.is_dataset_assigned():
            raise RuntimeError("Dataset must be assigned before loading partitions")
        if partition_id < 0 or partition_id >= self.num_partitions:
            raise ValueError(f"Invalid partition ID: {partition_id}")

        return self.dataset.select(self._partition_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    def __repr__(self) -> str:
        return (f"ClassPartitioner(num_partitions={self._num_partitions}, "
                f"seed={self._seed}, label_column='{self._label_column}')")


def _get_item_label(item, label_key='label'):
    """Retorna o rótulo de um item de dataset, tentando várias estratégias comuns:
        - Se for um dict, tenta item[label_key]
        - Se for uma tupla/lista, tenta item[1] (comum: (x, label)) ou item[0] se não houver item[1]
        - Caso contrário, tenta converter o item inteiro para int (pode ser o caso de datasets que retornam apenas o rótulo)
    """
    if isinstance(item, dict):
        return int(item[label_key])
    if isinstance(item, (list, tuple)):
        maybe_label = item[1] if len(item) > 1 else item[0]
        return int(maybe_label)
    return int(item)

def get_label_counts(dataset, label_key='label', max_samples=None) -> Counter:
    """Retorna um Counter com a contagem de amostras por rótulo em um dataset, tentando extrair rótulos usando várias estratégias comuns (ver _get_item_label).
       Se max_samples for fornecido, conta no máximo max_samples itens (útil para datasets muito grandes).
    """
    indices = None
    if isinstance(dataset, Subset):
        indices = dataset.indices
        underlying = dataset.dataset
    else:
        indices = None
        underlying = dataset

    counts = Counter()
    n = 0
    if indices is not None:
        for idx in indices:
            item = underlying[idx]
            counts[_get_item_label(item, label_key)] += 1
            n += 1
            if max_samples and n >= max_samples:
                break
    else:
        for item in dataset:
            counts[_get_item_label(item, label_key)] += 1
            n += 1
            if max_samples and n >= max_samples:
                break
    return counts

def choose_minority_labels(counts: Counter,
                           total_num_classes: int = 10,
                           method: str = 'topk',
                           k: Optional[int] = None,
                           threshold: Optional[int] = None,
                           ratio: Optional[float] = None) -> List[int]:
    """Seleciona rótulos minoritários com base em contagens de rótulos, usando um dos seguintes métodos:
    - 'topk': seleciona os k rótulos com as menores contagens (empates arbitrários)
    - 'threshold': seleciona rótulos com contagem ≤ threshold
    - 'ratio': seleciona rótulos com contagem ≤ ratio * max_count (ou seja, dentro de uma certa proporção do rótulo majoritário)
    """

    full_counts = {lbl: counts.get(lbl, 0) for lbl in range(total_num_classes)}

    if method == 'topk':
        if k is None:
            raise ValueError("k must be provided for topk method")
        return [l for l, _ in sorted(full_counts.items(), key=lambda kv: kv[1])][:k]
    elif method == 'threshold':
        if threshold is None:
            raise ValueError("threshold must be provided for threshold method")
        return [l for l, c in full_counts.items() if c <= threshold]
    elif method == 'ratio':
        if ratio is None:
            raise ValueError("ratio must be provided for ratio method")
        maxc = max(full_counts.values()) if full_counts else 0
        return [l for l, c in full_counts.items() if c <= ratio * maxc]
    else:
        raise ValueError(f"Unknown method '{method}'.")


def build_label_index_map(dataset, label_key='label') -> Dict[int, List[int]]:
    """Constrói um dicionário que mapeia cada rótulo para uma lista de índices correspondentes no dataset.
    """
    label_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset[idx]
        lbl = _get_item_label(item, label_key)
        label_to_indices[int(lbl)].append(idx)
    return label_to_indices

def sample_generated_indices_for_labels(gen_dataset,
                                        desired_labels: List[int],
                                        num_per_label: Dict[int,int],
                                        label_key='label',
                                        rng_seed=None) -> List[int]:
    """Retorna uma lista de índices do gen_dataset, amostrando num_per_label[l] índices para cada rótulo l.
       Se não houver amostras geradas suficientes para um rótulo, ele irá amostrar quantas estiverem disponíveis e avisar.
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    label_map = build_label_index_map(gen_dataset, label_key=label_key)
    chosen = []
    for lbl in desired_labels:
        available = label_map.get(lbl, [])
        need = num_per_label.get(lbl, 0)
        if need <= 0:
            continue
        if len(available) == 0:
            print(f"Warning: no generated samples for label {lbl}")
            continue
        if len(available) < need:
            print(f"Warning: requested {need} for label {lbl} but only {len(available)} available; taking all.")
            chosen.extend(available)
        else:
            chosen.extend(random.sample(available, need))
    return chosen

class _IndexWrappingDataset(torch.utils.data.Dataset):
    """Wrapper simples para criar um dataset que é um subconjunto de outro dataset, usando uma lista de índices.
    """
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = list(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.base[self.indices[idx]]
    

def unpack_batch(batch, image_key='image', label_key='label'):
    """
    Desempacota um batch de dados, tentando extrair imagens e rótulos usando várias estratégias comuns:
        - Se for um dict, tenta batch[image_key] e batch[label_key]
        - Se for uma tupla/lista, tenta batch[0] como imagens e batch[1] como rótulos (comum: (images, labels) ou ((images, ...), labels))
        - Caso contrário, levanta um erro (não suporta outros formatos de batch)
    """
    if isinstance(batch, dict):
        images = batch[image_key]
        labels = batch[label_key]
    elif isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            raise ValueError("Tuple batch with unexpected length")
    else:
        raise ValueError("Unsupported batch type: %s" % type(batch))
    return images, labels

class EmbeddingPairDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels, asset_col_name='image', label_col_name='label'):
        self.emb = embeddings
        self.lbl = labels
        self.asset_col_name = asset_col_name
        self.label_col_name = label_col_name
    def __len__(self):
        return self.emb.size(0)
    def __getitem__(self, idx):
        return {self.asset_col_name: self.emb[idx], self.label_col_name: int(self.lbl[idx])}
    
def augment_client_with_generated(client_train,
                                  gen_dataset,
                                  counts,
                                  label_key_client='label',
                                  label_key_gen='label',
                                  strategy='fill_to_max',
                                  fill_to: Optional[int] = None,
                                  k: Optional[int] = None,
                                  threshold: Optional[int] = None,
                                  ratio: Optional[float] = None,
                                  rng_seed: Optional[int] = None) -> Tuple[torch.utils.data.Dataset, Dict]:
    """
    Retorna um novo dataset que é a concatenação de client_train com um subconjunto de gen_dataset, onde o subconjunto é escolhido para "preencher" os rótulos 
    minoritários em client_train de acordo com a estratégia especificada.
    """

    if len(counts)==0:
        print("Warning: client_train has zero samples.")
    max_count = max(counts.values()) if counts else 0

    if strategy == 'fill_to_max':
        desired_labels = [l for l, c in counts.items() if c < max_count]
        per_label_target = {l: max_count for l in desired_labels}
    elif strategy == 'fill_to':
        if fill_to is None:
            raise ValueError("fill_to must be provided for 'fill_to' strategy")
        desired_labels = [l for l, c in counts.items() if c < fill_to]
        per_label_target = {l: fill_to for l in desired_labels}
    elif strategy == 'topk':
        if k is None or fill_to is None:
            raise ValueError("k and fill_to required for topk")
        desired_labels = choose_minority_labels(counts, method='topk', k=k)
        per_label_target = {l: fill_to for l in desired_labels}
    elif strategy == 'threshold':
        if threshold is None or fill_to is None:
            raise ValueError("threshold and fill_to required for threshold")
        desired_labels = choose_minority_labels(counts, method='threshold', threshold=threshold)
        per_label_target = {l: fill_to for l in desired_labels}
    elif strategy == 'ratio':
        if ratio is None or fill_to is None:
            raise ValueError("ratio and fill_to required for ratio")
        desired_labels = choose_minority_labels(counts, method='ratio', ratio=ratio)
        per_label_target = {l: fill_to for l in desired_labels}
    else:
        raise ValueError("Unknown strategy")

    need_per_label = {}
    for l in desired_labels:
        need = per_label_target[l] - counts.get(l, 0)
        if need > 0:
            need_per_label[l] = need

    chosen_gen_indices = sample_generated_indices_for_labels(
        gen_dataset,
        desired_labels=desired_labels,
        num_per_label=need_per_label,
        label_key=label_key_gen,
        rng_seed=rng_seed
    )

    gen_subset = _IndexWrappingDataset(gen_dataset, chosen_gen_indices) if chosen_gen_indices else None

    if gen_subset is None or len(gen_subset) == 0:
        combined = client_train
    else:
        combined = ConcatDataset([client_train, gen_subset])

    stats = {
        'client_counts': counts,
        'desired_labels': desired_labels,
        'need_per_label': need_per_label,
        'gen_selected_count': len(chosen_gen_indices)
    }
    return combined, stats


def plot_series(
    series: Mapping[str, Iterable[float]],
    *,
    series_styles: Mapping[str, Mapping[str, Any]] = None,
    subplot_groups: List[List[str]] = None,
    subplot_layout: Tuple[int, int] = None,
    subplot_margins: dict = None,
    legend_subplot_index: Union[int, List[int], str] = 'all',
    legend_loc: str = 'best',
    legend_fontsize: float = 10,
    legend_kwargs: Union[Mapping[str, Any], List[Mapping[str, Any]]] = None,
    title: Union[str, List[str]] = None,
    title_fontsize: float = None,
    row_suptitles: List[str] = None, 
    row_suptitle_fontsize: float = 14,
    figure_title: str = None,
    figure_title_fontsize: float = 16,
    figure_title_y: float = None,
    x_ticks: Union[List[float], List[List[float]]] = None,
    y_ticks: Union[List[float], List[List[float]]] = None,
    xtick_step: Union[int, List[int]] = 1,
    xtick_offset: int = 0,
    first_step_xtick: Union[int, List[int]] = None,
    tick_fontsize: float = None,
    num_xticks: Union[int, List[int]] = None,
    num_yticks: Union[int, List[int]] = None,
    hide_inner_ticks: bool = False,
    xlim: Union[tuple[float, float], List[tuple[float, float]]] = None,
    ylim: Union[tuple[float, float], List[tuple[float, float]]] = None,
    xlabel: Union[str, List[str]] = "Epochs",
    ylabel: Union[str, List[str]] = "Value",
    label_fontsize: float = None,
    row_labels: List[str] = None,
    row_label_fontsize: float = None,
    highlight: Mapping[str, Literal["max", "min", "both"]] = None,
    highlight_marker: str = "o",
    highlight_markersize: float = 4,
    highlight_color: str = None,
    highlight_text_size: int = 8,
    highlight_text_color: str = None,
    highlight_arrow_color: str = None,
    highlight_arrow_style: str = "->",
    highlight_arrow_linewidth: float = 1,
    highlight_text_offset_max: tuple[float, float] = (0.1, 0.2),
    highlight_text_offset_min: tuple[float, float] = (0.1, -0.2),
    highlight_style: Mapping[str, Mapping[str, Any]] = None,
    figsize: tuple[float, float] = (10, 5),
    hspace: float = None,
    vspace: float = None,
    save: bool = False,
    plot_name: str = "plot.pdf",
    level_markers: Union[dict, List[dict]] = None,
) -> None:
    """
    Plota séries temporais com uma variedade de opções de personalização.
    """
    if subplot_groups is None:
        subplot_groups = [list(series.keys())]

    num_plots = len(subplot_groups)

    nrows, ncols = 0, 0
    if subplot_layout:
        nrows, ncols = subplot_layout
        if nrows * ncols < num_plots:
            raise ValueError(f"Layout {subplot_layout} is too small for {num_plots} groups.")
    else:
        nrows, ncols = num_plots, 1

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    def get_setting(value, index):
        if isinstance(value, list):
            
            if value is x_ticks or value is y_ticks:
                if len(value) > 0 and isinstance(value[0], list):
                    return value[index] if index < len(value) else None
                else:
                    return value
            
            return value[index] if index < len(value) else None
        return value

    for i, (ax, group) in enumerate(zip(axes, subplot_groups)):

        row = i // ncols
        col = i % ncols
        is_bottom_row = (row == nrows - 1)
        is_left_col = (col == 0)

        n = 0
        if group:
            n = max(len(series.get(name, [])) for name in group)

        for name in group:
            if name not in series:
                continue
            ys = series[name]
            xs = range(len(ys))

            raw_style = series_styles.get(name, {}) if series_styles else {}
            style = raw_style.copy()
            plot_label = style.pop('label', name)

            line, = ax.plot(xs, ys, label=plot_label, **style)
            
            mode = highlight.get(name) if highlight else None
            base_color = style.get('color', line.get_color())
            mcolor = highlight_color or base_color

            current_highlight_style = highlight_style.get(name, {}) if highlight_style else {}

            if mode in ("max", "both"):
                i_max = max(range(len(ys)), key=lambda j: ys[j])
                x_coord_max = xs[i_max] if hasattr(xs, '__getitem__') else list(xs)[i_max]
                ax.plot(i_max, ys[i_max], marker=highlight_marker, markersize=highlight_markersize, color=mcolor)
                offset = current_highlight_style.get('highlight_offset_max', highlight_text_offset_max)
                text_position = (i_max + offset[0], ys[i_max] + offset[1])
                arrow_color = current_highlight_style.get('arrow_color', highlight_arrow_color or 'dimgrey')
                arrow_style = current_highlight_style.get('arrow_style', highlight_arrow_style)
                arrow_width = current_highlight_style.get('arrow_linewidth', highlight_arrow_linewidth)
                text_color = current_highlight_style.get('text_color', highlight_text_color or 'black')
                ax.annotate(f"{ys[i_max]:.2f}",
                            xy=(i_max, ys[i_max]), 
                            xytext=text_position,
                            arrowprops=dict(arrowstyle=arrow_style, color=arrow_color, linewidth=arrow_width),
                            fontsize=highlight_text_size, 
                            color=text_color,
                            va="bottom", 
                            ha="center")

            if mode in ("min", "both"):
                i_min = min(range(len(ys)), key=lambda j: ys[j])
                ax.plot(i_min, ys[i_min], marker=highlight_marker, markersize=highlight_markersize, color=mcolor)
                offset = current_highlight_style.get('highlight_offset_min', highlight_text_offset_min)
                text_position = (i_min + offset[0], ys[i_min] + offset[1])
                arrow_color = current_highlight_style.get('arrow_color', highlight_arrow_color or 'dimgrey')
                arrow_style = current_highlight_style.get('arrow_style', highlight_arrow_style)
                arrow_width = current_highlight_style.get('arrow_linewidth', highlight_arrow_linewidth)
                text_color = current_highlight_style.get('text_color', highlight_text_color or 'black')
                ax.annotate(f"{ys[i_min]:.2f}", 
                            xy=(i_min, ys[i_min]), 
                            xytext=text_position,
                            arrowprops=dict(arrowstyle=arrow_style, color=arrow_color, linewidth=arrow_width),
                            fontsize=highlight_text_size,
                            color=text_color,
                            va="top", 
                            ha="center")

        if n > 0:
            current_num_yticks = get_setting(num_yticks, i)
            current_y_ticks = get_setting(y_ticks, i)

            current_num_xticks = get_setting(num_xticks, i)
            current_x_ticks = get_setting(x_ticks, i)

            current_first_step_xtick = get_setting(first_step_xtick, i)
            current_xtick_step = get_setting(xtick_step, i)

            if current_num_yticks or current_y_ticks:
                if current_num_yticks:
                    current_ylim = get_setting(ylim, i)
                    if current_ylim:
                        min_y, max_y = current_ylim
                    else:
                        min_y = float('inf')
                        max_y = float('-inf')
                        for name in group:
                            if name in series and len(series[name]) > 0:
                                min_y = min(min_y, min(series[name]))
                                max_y = max(max_y, max(series[name]))

                        if min_y == float('inf') or max_y == float('-inf'):
                            min_y, max_y = 0, 1.0
                        
                        min_y = math.floor(min_y * 10) / 10
                        max_y = math.ceil(max_y * 10) / 10

                    yticks = np.linspace(min_y, max_y, current_num_yticks)

                    yticks = np.unique(yticks)
                else:
                    yticks = current_y_ticks
                ax.set_yticks(yticks)
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))

            if current_x_ticks is not None:
                ax.set_xticks(current_x_ticks)
            elif current_num_xticks:
                xticks = np.linspace(1, n, current_num_xticks)
                ax.set_xticks(xticks.astype(int))
            elif current_first_step_xtick is not None:
                labels = [1]
                step = current_xtick_step if current_xtick_step is not None else 1
                next_label = 1 + current_first_step_xtick
                while next_label <= n:
                    labels.append(next_label)
                    next_label += step
                positions = [lbl - 1 for lbl in labels]
                labels = [lbl + xtick_offset for lbl in labels]
                ax.set_xticks(positions, labels)
            elif current_xtick_step is not None and current_xtick_step > 0:
                positions = list(range(0, n, current_xtick_step))
                labels = [pos + 1 + xtick_offset for pos in positions]
                ax.set_xticks(positions, labels)
            
        if hide_inner_ticks:
            if not is_bottom_row:
                ax.set_xticklabels([])
            if not is_left_col:
                ax.set_yticklabels([])

        if num_xticks and xtick_offset != 0 and n > 0 and x_ticks is None:
            fig.canvas.draw()
            current_ticks = ax.get_xticks()
            new_labels = [int(tick) + xtick_offset for tick in current_ticks]
            ax.set_xticklabels(new_labels)

        if tick_fontsize:
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        x_text = get_setting(xlabel, i)
        if isinstance(xlabel, list):
            ax.set_xlabel(x_text, fontsize=label_fontsize)
        elif is_bottom_row:
            ax.set_xlabel(x_text, fontsize=label_fontsize)

        y_text = get_setting(ylabel, i)
        if isinstance(ylabel, list):
            ax.set_ylabel(y_text, fontsize=label_fontsize)
        elif is_left_col:
            ax.set_ylabel(y_text, fontsize=label_fontsize)

        ax.set_title(get_setting(title, i), fontsize=title_fontsize)

        show_legend = False
        if legend_subplot_index == 'all':
            show_legend = True
        elif isinstance(legend_subplot_index, list):
            if i in legend_subplot_index:
                show_legend = True
        elif i == legend_subplot_index:
            show_legend = True
        
        if show_legend:
            base_kwargs = {'loc': legend_loc, 'fontsize': legend_fontsize}
            current_kwargs = get_setting(legend_kwargs, i)
            if current_kwargs:
                base_kwargs.update(current_kwargs)
            ax.legend(**base_kwargs)

        current_xlim = get_setting(xlim, i)
        if current_xlim:
            ax.set_xlim(*current_xlim)
        elif n > 0:
            if x_ticks and not isinstance(x_ticks, dict):
                 ax.set_xlim(min(x_ticks), max(x_ticks))
            else:
                 ax.set_xlim(0, n)

        current_ylim = get_setting(ylim, i)
        if current_ylim:
            ax.set_ylim(*current_ylim)

        if row_labels and (col == ncols - 1):
            if row < len(row_labels) and row_labels[row]:
                ax.text(
                    1.05, 0.5, row_labels[row],  
                    transform=ax.transAxes,      
                    rotation=270,               
                    ha='left', 
                    va='center',
                    fontsize=row_label_fontsize,
                    fontweight='bold'           
                )

        middle_col_index = ncols // 2
        
        if row_suptitles and (col == middle_col_index):
            if row < len(row_suptitles) and row_suptitles[row]:
                ax.text(
                    0.5, 1.2, row_suptitles[row],
                    transform=ax.transAxes,
                    ha='center', 
                    va='bottom',
                    fontsize=row_suptitle_fontsize,
                    fontweight='bold'
                )

        current_markers = get_setting(level_markers, i)

        if current_markers:
            for label, x_pos in current_markers.items():
                ax.axvline(
                    x=x_pos, 
                    color='gray', 
                    linestyle='--', 
                    linewidth=2, 
                    alpha=0.8,
                    zorder=0 
                )

                ax.text(
                    x=x_pos, 
                    y=0.05, 
                    s=label, 
                    transform=ax.get_xaxis_transform(), 
                    rotation=90,      
                    ha='right',       
                    va='bottom',
                    fontsize=18,
                    color='dimgrey',
                    fontweight='bold'
                )

    if figure_title:
        fig.suptitle(
            figure_title, 
            fontsize=figure_title_fontsize,
            y=figure_title_y or 0.98
        )

    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)

    if hspace is not None or vspace is not None or subplot_margins:
        margins = subplot_margins or {}
        plt.subplots_adjust(
            hspace=hspace or 0.3,  
            wspace=vspace or 0.2, 
            left=margins.get('left', 0.1),
            right=margins.get('right', 0.9),
            top=margins.get('top', 0.9),
            bottom=margins.get('bottom', 0.1)
        )
    else:
        fig.tight_layout()

    shift_amount = 0.04 

    for i, ax in enumerate(axes):
        row = i // ncols

        if row != 0:
            if i in (9,15):
                shift_amount += 0.04
            pos = ax.get_position()
            
            new_pos = [pos.x0, pos.y0 + shift_amount, pos.width, pos.height]
            
            ax.set_position(new_pos)

    if save:
        plt.savefig(plot_name)
    plt.show()

def plot_by_marker(ax, x_data, y_data, c_data, m_data, title):
    """
    Função auxiliar para plotar dados agrupando-os por marcador.
    """
    x_arr = np.array(x_data)
    y_arr = np.array(y_data)
    c_arr = np.array(c_data)
    m_arr = np.array(m_data)

    unique_markers = np.unique(m_arr)
    
    for m in unique_markers:
        mask = (m_arr == m)
        
        ax.scatter(
            x_arr[mask], 
            y_arr[mask], 
            c=c_arr[mask], 
            marker=m,
            s=340, 
            alpha=0.9
        )
    
    ax.set_title(title, fontsize=22)
    ax.set_xlabel("Acurácia Máxima", fontsize=20) 
    ax.set_ylabel("Tráfego (GB)", fontsize=20)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, linestyle='--', alpha=0.4)

def calculate_times_and_accs(exp_data, is_baseline):
    """
    Retorna listas de tempos e acurácias para plotagem, calculando os tempos de forma diferente para o caso baseline (cumulative sum simples) e para o caso 
    FLEG (acumulando tempos de época e adicionando os gaps de treinamento do GAN conforme indicado por accuracy_transition e time_level).
    """
    accs = exp_data.get('net_acc', [])
    epoch_times = exp_data.get('time_epoch_classifier', [])
    
    # --- Caso 1: Baseline ---
    if is_baseline:
        times = np.cumsum(epoch_times).tolist()
        return times, accs

    # --- Caso 2: FLEG ---
    transitions = exp_data.get('accuracy_transition', [])
    time_levels = exp_data.get('time_level', [])
    
    calculated_times = []
    current_cumulative_time = 0
    current_epoch_idx = 0
    
    for i, target_acc in enumerate(transitions):
        found_idx = -1
        for k in range(current_epoch_idx, len(accs)):
            if math.isclose(accs[k], target_acc, rel_tol=1e-9):
                found_idx = k
                break
            
        level_classifier_time = 0
        for k in range(current_epoch_idx, found_idx + 1):
            t = epoch_times[k]
            level_classifier_time += t
            current_cumulative_time += t
            calculated_times.append(current_cumulative_time)
            
        if i < len(time_levels)-1:
            total_level_duration = time_levels[i]
            gan_time = total_level_duration - level_classifier_time
            
            gan_time = max(0, gan_time)
            
            current_cumulative_time += gan_time
            
        current_epoch_idx = found_idx + 1
        
    for k in range(current_epoch_idx, len(accs)):
        t = epoch_times[k]
        current_cumulative_time += t
        calculated_times.append(current_cumulative_time)
    
    return calculated_times, accs

def parse_int_or_list(value):
    if "," not in value:
        return int(value)

    return [int(v.strip()) for v in value.split(",")]

def build_client_chunks(client_datasets, num_chunks_lvl, seed):
    client_chunks = []

    for train_partition in client_datasets:
        dts = train_partition["train"]
        n = len(dts)

        indices = list(range(n))
        random.seed(seed)
        random.shuffle(indices)

        chunk_size = math.ceil(n / num_chunks_lvl)

        chunks = []
        for i in range(num_chunks_lvl):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            chunk_indices = indices[start:end]
            chunks.append(Subset(dts, chunk_indices))

        client_chunks.append(chunks)

    return client_chunks

def aggregate_pytorch_models(client_models_data):
    """
    Agrega os pesos dos modelos dos clientes usando média ponderada (FedAvg).
    
    Argumentos:
        client_models_data: Lista de tuplas (state_dict, num_samples)
                            Ex: [(state_dict_cliente1, 150), (state_dict_cliente2, 300)]
    Retorna um novo state_dict global agregado.
    """
    # 1. Calcula o total de amostras de todos os clientes
    total_samples = sum(num_samples for _, num_samples in client_models_data)
    
    # 2. Cria um dicionário vazio para armazenar os pesos globais
    global_state_dict = {}
    
    # Pegamos as chaves (nomes das camadas) do primeiro cliente como referência
    first_client_dict = client_models_data[0][0]
    
    # 3. Itera sobre cada camada do modelo
    for key in first_client_dict.keys():
        # Inicializa um tensor de zeros com o mesmo formato da camada, no mesmo device
        global_state_dict[key] = torch.zeros_like(first_client_dict[key])
        
        # Soma a contribuição ponderada de cada cliente para esta camada
        for client_state_dict, num_samples in client_models_data:
            weight = num_samples / total_samples
            global_state_dict[key] += client_state_dict[key] * weight
            
    return global_state_dict
