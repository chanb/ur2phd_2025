import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from torch.utils.data import (
    DataLoader,
    Dataset,
)
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import dill
import numpy as np
import timeit
import torch

import src.datasets as datasets
import src.models as models

from src.constants import *


class NextTokenLearner:
    """
    A learner for training models to perform next-token prediction.
    """
    def __init__(
        self,
        config: SimpleNamespace,
    ):
        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._dataset, self._dataset_loader = self.make_dataset_and_loader(
            config.dataset_config.train
        )

        if hasattr(config.dataset_config, "validation"):
            self._val_dataset, self._val_dataset_loader = self.make_dataset_and_loader(
                config.dataset_config.validation
            )

        self._model, self._optimizer = self.make_model_and_optimizer()

        # Load pretrained model if specified
        if getattr(self.config, "pretrained_model_path", None):
            state_dict = torch.load(
                self.config.pretrained_model_path,
                map_location=self.device,
            )
            self.load_state_dict(state_dict)

    @property
    def config(self) -> SimpleNamespace:
        """
        Configuration.
        """
        return self._config

    @property
    def device(self) -> torch.device:
        """
        Device.
        """
        return self._device

    @property
    def dataset(self) -> Dataset:
        """
        Dataset.
        """
        return self._dataset

    @property
    def model(self) -> torch.nn.Module:
        """
        Model.
        """
        return self._model

    @property
    def optimizer(self) -> torch.nn.Module:
        """
        Optimizer.
        """
        return self._optimizer

    @property
    def state(self) -> Dict:
        """
        Model states and optimizer states.
        """
        return {
            CONST_MODEL_STATE: self.model.state_dict(),
            CONST_OPT_STATE: self.optimizer.state_dict(),
        }
    
    def close(self):
        del self._dataset, self._dataset_loader

    def load_state_dict(self, state_dict: Dict):
        """
        Loads the states for both the model and optimizer.

        :param state_dict: the state dictionary
        :type state_dict: Dict
        """
        self.model.load_state_dict(state_dict[CONST_MODEL_STATE])
        self.optimizer.load_state_dict(state_dict[CONST_OPT_STATE])

    def make_dataset_and_loader(
        self,
        dataset_config: SimpleNamespace,
    ) -> Tuple[Dataset, DataLoader]:
        """
        Constructs the dataset and dataset loader.

        :param dataset_config: the dataset configuration
        :type dataset_config: SimpleNamespace
        :return: the dataset and dataset loader
        :rtype: Tuple[Dataset, DataLoader]

        """
        dataset_class = getattr(
            datasets,
            dataset_config.dataset_name,
        )
        dataset = dataset_class(
            **vars(dataset_config.kwargs),
        )

        dataset_loader = DataLoader(
            dataset=dataset,
            batch_size=dataset_config.batch_size,
            shuffle=getattr(dataset_config, "shuffle", True),
            drop_last=getattr(dataset_config, "drop_last", True),
            num_workers=getattr(dataset_config, "num_workers", 0),
        )
        return dataset, dataset_loader

    def make_model_and_optimizer(self) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        """
        Constructs the model and optimizer.

        :return: the model and optimizer
        :rtype: Tuple[torch.nn.Module, torch.optim.Optimizer]

        """
        model_class = getattr(models, self.config.model_config.architecture)
        model_kwargs = vars(self.config.model_config.kwargs)
        model_kwargs.update(
            vocab_size=self.dataset.vocab_size
        )
        model = model_class(
            **model_kwargs,
        )

        optimizer_class = getattr(torch.optim, self.config.optimizer_config.optimizer)
        optimizer = optimizer_class(
            model.parameters(),
            **vars(self.config.optimizer_config.kwargs)
        )

        return model.to(self.device), optimizer

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Any:
        """
        A single training step.

        :param batch: the batch
        :type batch: Dict[str, torch.Tensor]
        :return: the auxiliary information
        :rtype: Any

        """
        # TODO: Implement next-token prediction
        return {CONST_AGG_LOSS: torch.tensor(0.0, device=self.device)}

    def update(self, epoch: int, *args, **kwargs) -> Dict[str, Any]:
        """
        Updates the model.

        :param epoch: the epoch
        :type epoch: int
        :return: the update information
        :rtype: Dict[str, Any]

        """
        auxes = []
        total_sample_time = 0
        total_update_time = 0

        tic = timeit.default_timer()
        for batch_i, batch in enumerate(self._dataset_loader):
            # Put everything into the correct device
            batch = {
                k: v.to(self.device) for k, v in batch.items()
            }
            total_sample_time += timeit.default_timer() - tic

            tic = timeit.default_timer()
            aux = self.train_step(batch)
            total_update_time += timeit.default_timer() - tic
            assert np.isfinite(aux[CONST_AGG_LOSS].item()), f"Loss became NaN\naux: {aux}"

            auxes.append(aux)

            # This keeps track of batch sampling time
            tic = timeit.default_timer()

        log = {
            f"time/{CONST_SAMPLE_TIME}": total_sample_time,
            f"time/{CONST_UPDATE_TIME}": total_update_time,
            # TODO: Insert logging information
        }
        return log

    def validation(self, epoch: int) -> Dict[str, Any]:
        """
        Perform validation.

        :param epoch: the epoch
        :type epoch: int
        :return: the auxiliary information
        :rtype: Dict[str, Any]

        """

        # TODO: Include validation logic
        return {}
