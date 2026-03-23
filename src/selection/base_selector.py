# src/selection/base_selector.py
from abc import ABC, abstractmethod
import torch


class BaseChannelSelector(ABC):
    """Base class for all channel selection methods"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def select(self, model, dataloader, k: int, num_samples: int = 100, 
               device: str = 'cuda'):
        """
        Select top-k channels based on specific criterion
        
        Args:
            model: Target model with conv1 layer
            dataloader: Data loader for computing channel importance
            k: Number of channels to select
            num_samples: Number of samples to use for selection
            device: Computation device
            
        Returns:
            selected_channels: Tensor of shape (k,) containing channel indices
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"