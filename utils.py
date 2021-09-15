import json
import torch
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt

def convert_token_to_idx(tokens, token2idx):
    for token in tokens:
        yield [token2idx[t] for t in token]
    return


def convert_label_to_idx(labels, label2idx):
    for label in labels:
        yield label2idx[label]
    return


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


class Config:

    def __init__(self, json_path_or_dict: Union[str, dict]) -> None:

        if isinstance(json_path_or_dict, dict):
            self.__dict__.update(json_path_or_dict)
        else:
            with open(json_path_or_dict, mode='r') as io:
                params = json.loads(io.read())
            self.__dict__.update(params)

    def save(self, json_path: Union[str, Path]) -> None:

        with open(json_path, mode='w') as io:
            json.dump(self.__dict__, io, indent=4)

    def update(self, json_path_or_dict) -> None:

        if isinstance(json_path_or_dict, dict):
            self.__dict__.update(json_path_or_dict)
        else:
            with open(json_path_or_dict, mode='r') as io:
                params = json.loads(io.read())
            self.__dict__update(params)

    @property
    def dict(self) -> dict:
        return self.__dict__


class CheckpointManager:

    def __init__(self, model_dir: Union[str, Path]) -> None:

        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)

        if not model_dir.exists():
            model_dir.mkdir(parents=True)

        self._model_dir = model_dir

    def save_checkpoint(self, state: dict, filename: str) -> None:
        torch.save(state, self._model_dir / filename)

    def load_checkpoint(self, filename: str, device: torch.device = None) -> dict:

        device = device or (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        state = torch.load(self._model_dir / filename, map_location=device)
        return state


class SummaryManager:

    def __init__(self, model_dir: Union[str, Path]) -> None:
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)

        if not model_dir.exists():
            model_dir.mkdir(parents=True)

        self._model_dir = model_dir
        self._summary = {}

    def save(self, filename: str) -> None:

        with open(self._model_dir / filename, mode='w') as io:
            json.dump(self._summary, io, indent=4)

    def load(self, filename) -> None:

        with open(self._model_dir / filename, mode='r') as io:
            metric = json.loads(io.read())
        self.update(metric)

    def update(self, summary: dict) -> None:

        self._summary.update(summary)

    def reset(self) -> None:

        self._summary = {}

    @property
    def summary(self):
        return self._summary_summary