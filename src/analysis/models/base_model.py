# src/analysis/models/base_model.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union


class Model(ABC):
    """
    Abstract base class for all models in the lysis analysis framework.
    """

    def __init__(self, name: str):
        self.name = name
        self.params = {}
        self.fitted = False
        self.training_data = None

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> bool:
        """
        Fit the model to the given data.

        Args:
            data: DataFrame containing the data to fit the model to

        Returns:
            bool: True if fitting was successful, False otherwise
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using the fitted model.

        Args:
            data: DataFrame to generate predictions for

        Returns:
            DataFrame with predictions added
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Return the current model parameters.

        Returns:
            Dict of parameter names and values
        """
        return self.params.copy()

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set model parameters manually (without fitting).

        Args:
            params: Dict of parameter names and values
        """
        for key, value in params.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise ValueError(f"Unknown parameter '{key}' for {self.name} model")

    def has_required_params(self) -> bool:
        """
        Check if all required parameters are set and valid.

        Returns:
            bool: True if all required parameters are set, False otherwise
        """
        return all(pd.notna(value) for value in self.params.values())

    def clone(self):
        """
        Create a copy of this model with the same parameters.

        Returns:
            A new instance of the same model with identical parameters
        """
        # Default implementation that children can override if needed
        new_model = self.__class__(self.name)
        new_model.set_params(self.get_params())
        new_model.fitted = self.fitted
        return new_model