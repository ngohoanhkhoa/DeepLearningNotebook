from os.path import isfile
from typing import Any, Dict
import torch
from kedro.io import AbstractDataSet


class TorchModel(AbstractDataSet):

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)

    def __init__(
        self,
        filepath: str,
    ) -> None:
        self._filepath = filepath

    def _exists(self) -> bool:
        return isfile(self._filepath)

    def _save(self, model) -> None:
        torch.save(model, self._filepath)

    def _load(self):
        model = torch.load(self._filepath)
        return model
