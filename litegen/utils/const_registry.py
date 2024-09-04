from typing import Any, Dict, List, Tuple, Type

import torch


class ConstRegistry:
    """
    class for constant registry
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConstRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._support_optimizers: List[Type] = []
        self._support_datasets: List[Type] = []
        self._support_attention: List[Type] = []
        self._sequence_parallel_attn_processor_convert_map: Dict[Type, Dict[Type, Type]] = {}

        self._load_optimizers()
        self._load_datasets()
        self._load_attention()

    def _load_optimizers(self):
        self._support_optimizers.append(torch.optim.Adam)
        self._support_optimizers.append(torch.optim.AdamW)

        self._support_optimizers = tuple(self._support_optimizers)

    def _load_datasets(self):
        self.support_datasets.append(torch.utils.data.Dataset)
        try:
            from datasets import arrow_dataset  # huggingface datasets

            self._support_datasets.append(arrow_dataset.Dataset)
        except ImportError:
            pass

        self._support_datasets = tuple(self._support_datasets)

    def _load_attention(self):
        try:
            from models.vid_sd3.sparse_attention import (
                Attention,
                SparseAttnProcessor,
                SparseAttnProcessorSP,
            )

            self._support_attention.append(Attention)
            self._sequence_parallel_attn_processor_convert_map[Attention] = {SparseAttnProcessor: SparseAttnProcessorSP}
        except ImportError:
            pass

        self._support_attention = tuple(self._support_attention)

    @property
    def support_optimizers(self) -> Tuple[Type]:
        return self._support_optimizers

    @property
    def support_datasets(self) -> Tuple[Type]:
        return self._support_datasets

    @property
    def support_attention(self) -> Tuple[Type]:
        return self._support_attention

    @property
    def sequence_parallel_attn_processor_convert_map(self) -> Dict[Type, Dict[Type, Type]]:
        return self._sequence_parallel_attn_processor_convert_map

    def add_optimizer(self, optimizer: Type[torch.optim.Optimizer]) -> None:
        if optimizer not in self._support_optimizers:
            self._support_optimizers += (optimizer,)

    def add_dataset(self, dataset: Type[Any]) -> None:
        if dataset not in self._datasets:
            self._support_datasets += (dataset,)

    def add_attention(self, attention: Type[Any]) -> None:
        if attention not in self._support_attention:
            self._support_attention += (attention,)

    def add_attn_processor_convert(self, attention: Any, processor: Any, processor_sp: Any) -> None:
        if attention not in self._sequence_parallel_attn_processor_convert_map:
            self._sequence_parallel_attn_processor_convert_map[attention] = {}
        self._sequence_parallel_attn_processor_convert_map[attention][processor] = processor_sp
