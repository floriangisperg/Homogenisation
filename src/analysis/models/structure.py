# src/analysis/models/structure.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List


class BaseModel(ABC):
    """Base abstract class for all models."""

    def __init__(self, name: str):
        self.name = name
        self.params = {}
        self.fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> bool:
        """Fit model parameters to data."""
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions."""
        pass

    def has_required_params(self) -> bool:
        """Check if all parameters are set."""
        return all(pd.notna(value) for value in self.params.values())


class IntactFractionModel(BaseModel):
    """Base class for intact fraction models."""

    def __init__(self, name: str = "intact_fraction_model"):
        super().__init__(name)
        self.params = {
            "k": np.nan,  # Lysis coefficient
            "alpha": np.nan  # Pressure exponent
        }

    def predict_intact_fraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict intact fraction based on cumulative dose."""
        # Implementation goes here
        pass


class DNAModel(BaseModel):
    """Base class for DNA concentration models."""

    def __init__(self, name: str = "dna_model"):
        super().__init__(name)
        self.params = {
            "C_release_fresh": np.nan,  # DNA release coefficient for fresh biomass
            "C_release_frozen": np.nan  # DNA release coefficient for frozen biomass
        }