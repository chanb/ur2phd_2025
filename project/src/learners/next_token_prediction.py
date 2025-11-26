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

import numpy as np
import timeit
import torch

import src.datasets as datasets
import src.models as models

from src.constants import *
from src.utils import parse_dict


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

        self._val_data = dict()
        if hasattr(config.dataset_config, "validation"):
            for val_config in config.dataset_config.validation:
                val_config = parse_dict(val_config)
                curr_val_dataset, curr_val_dataset_loader = self.make_dataset_and_loader(
                    val_config,
                )
                self._val_data[val_config.name] = {
                    "dataset": curr_val_dataset,
                    "loader": curr_val_dataset_loader,
                    "max_decode_len": getattr(val_config, "max_decode_len", None),
                }

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

    def train_step(self, batch: Dict[str, torch.Tensor], train: bool) -> Any:
        """
        A single training step.
        The learner should update model parameters based on the sampled batch here.

        :param batch: the batch
        :param train: whether in training mode
        :type batch: Dict[str, torch.Tensor]
        :type train: bool
        :return: the auxiliary information
        :rtype: Any

        """
        targets = batch["target"]  # (batch_size, seq_len)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        # TODO: We should really ignore the questions for NTP loss computation.
        with torch.set_grad_enabled(train):
            output_dict = self.model(batch)
            preds = output_dict["output"]  # (batch_size, seq_len, vocab_size)
            loss = loss_fn(
                preds.view(-1, self.dataset.vocab_size),
                targets.view(-1),
            )

            loss = loss.view(targets.shape)  # (batch_size, seq_len)
            loss_mean = loss.mean()

        if train:
            # Take gradient step
            self.optimizer.zero_grad()
            loss_mean.backward()
            self.optimizer.step()

        acc = (
            preds.argmax(dim=-1) == targets
        ).float().detach().cpu()

        # TODO: We should include gradient norm for logging.
        return {
            CONST_AGG_LOSS: loss_mean.detach().cpu(),
            CONST_AGG_ACCURACY: acc.mean(),
            CONST_LOSS_PER_CONTEXT: {
                f"{CONST_LOSS}-context_{context_i}": loss[:, context_i].mean().detach().cpu()
                for context_i in range(loss.shape[1])
            },
            CONST_ACCURACY_PER_CONTEXT: {
                f"{CONST_ACCURACY}-context_{context_i}": acc[:, context_i].mean()
                for context_i in range(acc.shape[1])
            },
        }

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
        self.model.train()
        for batch_i, batch in enumerate(self._dataset_loader):
            # Put everything into the correct device
            batch = {
                k: v.to(self.device) for k, v in batch.items()
            }
            total_sample_time += timeit.default_timer() - tic

            tic = timeit.default_timer()
            aux = self.train_step(batch, train=True)
            total_update_time += timeit.default_timer() - tic
            assert np.isfinite(aux[CONST_AGG_LOSS].item()), f"Loss became NaN\naux: {aux}"

            auxes.append(aux)

            # This keeps track of batch sampling time
            tic = timeit.default_timer()

        auxes = torch.utils._pytree.tree_map(
            lambda *args: np.mean([np.asarray(el) for el in args]),
            *auxes,
        )

        log = {
            f"time/{CONST_SAMPLE_TIME}": total_sample_time,
            f"time/{CONST_UPDATE_TIME}": total_update_time,
            f"train/{CONST_AGG_LOSS}": auxes[CONST_AGG_LOSS],
            **{
                f"train-{CONST_LOSS_PER_CONTEXT}/{k}": v
                for k, v in auxes[CONST_LOSS_PER_CONTEXT].items()
            },
            f"train/{CONST_AGG_ACCURACY}": auxes[CONST_AGG_ACCURACY],
            **{
                f"train-{CONST_ACCURACY_PER_CONTEXT}/{k}": v
                for k, v in auxes[CONST_ACCURACY_PER_CONTEXT].items()
            },
        }
        return log

    def rollout(
        self,
        batch: Dict[str, torch.Tensor],
        max_length: int,
        eos_token: int,
        greedy_decoding: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform autoregressive prediction upon receiving the question.

        :param batch: the batch
        :param max_length: the maximum length to decode
        :param greedy_decoding: whether to use greedy decoding
        :type batch: Dict[str, torch.Tensor]
        :type max_length: int
        :type greedy_decoding: bool
        :return: the auxiliary information
        :rtype: Dict[str, Any]
        """

        with torch.no_grad():
            curr_state = self.model.init_state(batch)
            curr_input = batch["input"][:, [0]]

            dones = torch.zeros(
                curr_input.size(0),
                dtype=torch.bool,
                device=self.device,
            )
            outputs = [dones.unsqueeze(1).cpu()]
            max_prompt_length = batch["input"].size(1) - 1
            for step_i in range(max_length):
                output_dict = self.model({
                    "input": curr_input,
                    "state": curr_state,
                })
                preds = output_dict["output"]  # (batch_size, seq_len, vocab_size)

                if greedy_decoding:
                    next_token = preds[:, -1, :].argmax(dim=-1)  # (batch_size,)
                else:
                    raise ValueError("Only greedy decoding is supported currently.")

                # Set next token to the ground truth if still in question part
                next_token = torch.where(
                    step_i + 1 < batch["question_len"],
                    batch["input"][:, min(step_i + 1, max_prompt_length)],
                    next_token,
                )

                dones = torch.logical_or(
                    dones,
                    next_token == eos_token,
                )

                next_token = torch.where(
                    dones,
                    torch.full_like(next_token, eos_token),
                    next_token,
                )

                # Prepare for next step
                curr_input = next_token.unsqueeze(1)  # (batch_size, 1)
                curr_state = output_dict["state"]
                outputs.append(next_token.unsqueeze(1).cpu())
            outputs = torch.cat(outputs, dim=1)  # (batch_size, max_length)

        rollout_acc = 0.0
        for output_seq, target_seq, question_len, answer_len in zip(
            outputs,
            batch["target"].cpu(),
            batch["question_len"].cpu(),
            batch["answer_len"].cpu(),
        ):
            target = "".join([
                str(token.item())
                for token in target_seq[question_len:question_len + answer_len]
            ]) + f"{eos_token}"
            pred = "".join([
                str(token.item())
                for token in output_seq[question_len:]
            ])
            acc = int(target in pred)
            rollout_acc += acc / batch["input"].size(0)

        return {
            f"{CONST_ROLLOUT_ACCURACY}": rollout_acc,
        }

    def validation(self, epoch: int) -> Dict[str, Any]:
        """
        Perform validation.

        :param epoch: the epoch
        :type epoch: int
        :return: the auxiliary information
        :rtype: Dict[str, Any]

        """

        self.model.eval()
        all_logs = dict()
        for val_data_name, val_data in self._val_data.items():
            val_dataset_loader = val_data["loader"]

            auxes = []
            total_sample_time = 0
            total_update_time = 0

            tic = timeit.default_timer()
            for batch_i, batch in enumerate(val_dataset_loader):
                # Put everything into the correct device
                batch = {
                    k: v.to(self.device) for k, v in batch.items()
                }
                total_sample_time += timeit.default_timer() - tic

                tic = timeit.default_timer()
                aux = self.train_step(batch, train=False)
                if val_data["max_decode_len"] is not None:
                    aux.update(
                        self.rollout(
                            batch=batch,
                            max_length=val_data["max_decode_len"],
                            eos_token=val_data["dataset"].eos_token,
                            greedy_decoding=True,
                        )
                    )
                total_update_time += timeit.default_timer() - tic

                auxes.append(aux)

                # This keeps track of batch sampling time
                tic = timeit.default_timer()

            auxes = torch.utils._pytree.tree_map(
                lambda *args: np.mean([np.asarray(el) for el in args]),
                *auxes,
            )

            log = {
                f"time/val_{val_data_name}_{CONST_SAMPLE_TIME}": total_sample_time,
                f"time/val_{val_data_name}_{CONST_UPDATE_TIME}": total_update_time,
                f"train/val_{val_data_name}_{CONST_AGG_LOSS}": auxes[CONST_AGG_LOSS],
                f"train/val_{val_data_name}_{CONST_AGG_ACCURACY}": auxes[CONST_AGG_ACCURACY],
                f"train/val_{val_data_name}_{CONST_ROLLOUT_ACCURACY}": auxes[CONST_ROLLOUT_ACCURACY],
                **{
                    f"val-{val_data_name}-{CONST_LOSS_PER_CONTEXT}/{k}": v
                    for k, v in auxes[CONST_LOSS_PER_CONTEXT].items()
                },
                **{
                    f"val-{val_data_name}-{CONST_ACCURACY_PER_CONTEXT}/{k}": v
                    for k, v in auxes[CONST_ACCURACY_PER_CONTEXT].items()
                },
            }
            all_logs.update(log)
        return all_logs
