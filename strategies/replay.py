from typing import Optional, TYPE_CHECKING
from avalanche.benchmarks.utils import make_tensor_classification_dataset as MakeTensor
import numpy as np
import copy
import gc
import sys
from torch.utils.data import Dataset
from avalanche.benchmarks.utils import concat_classification_datasets
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
    ReservoirSamplingBuffer,
)
import torch
from itertools import chain
from avalanche.benchmarks.utils.data import AvalancheDataset
if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate
class Buffer(Dataset):
    def __init__(self, features, targets, task_ids):
        self.features = features
        self.targets = targets
        self.task_ids = task_ids
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.task_ids[idx]

class CustomSamplingBuffer(ReservoirSamplingBuffer):
    def __init__(self, max_size: int):
        super().__init__(max_size)
        self._buffer_weights = torch.zeros(0)

    def update_from_dataset(self, new_data: AvalancheDataset):
        """Update the buffer using the given dataset.
        :param new_data:
        :return:
        """
        new_weights = torch.rand(len(new_data))

        cat_weights = torch.cat([new_weights, self._buffer_weights])
        cat_data = new_data.concat(self.buffer)
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        buffer_idxs = sorted_idxs[: self.max_size]
        self.buffer = cat_data.subset(buffer_idxs)
        self._buffer_weights = sorted_weights[: self.max_size]
class CustomBuffer(ExperienceBalancedBuffer):
    def __init__(
        self, max_size: int, adaptive_size: bool = True, num_experiences=None
    ):
        super().__init__(max_size, adaptive_size, num_experiences)
    def update(self, strategy: "SupervisedTemplate", dataset, **kwargs):
        new_data = dataset
        num_exps = strategy.clock.train_exp_counter + 1
        lens = self.get_group_lengths(num_exps)

        new_buffer = ReservoirSamplingBuffer(lens[-1])
        new_buffer.update_from_dataset(new_data)
        self.buffer_groups[num_exps - 1] = new_buffer
        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(strategy, ll)

class CustomDataLoader(ReplayDataLoader):
    def __init__(
        self,
        data: AvalancheDataset,
        memory: Optional[AvalancheDataset] = None,
        oversample_small_tasks: bool = False,
        batch_size: int = 32,
        batch_size_mem: int = 32,
        task_balanced_dataloader: bool = False,
        distributed_sampling: bool = True,
        **kwargs
    ):
        super().__init__(data, memory, oversample_small_tasks, batch_size,
                batch_size_mem, task_balanced_dataloader, distributed_sampling, **kwargs)
        self.collate_fn = kwargs['collate_fn']
    
    def __iter__(self):
        loader_data, sampler_data = self._create_loaders_and_samplers(
            self.data, self.data_batch_sizes
        )
        loader_memory, sampler_memory = self._create_loaders_and_samplers(
            self.memory, self.memory_batch_sizes
        )

        iter_data_dataloaders = {}
        iter_buffer_dataloaders = {}

        for t in loader_data.keys():
            iter_data_dataloaders[t] = iter(loader_data[t])
        for t in loader_memory.keys():
            iter_buffer_dataloaders[t] = iter(loader_memory[t])

        max_len = max(
            [
                len(d)
                for d in chain(
                    loader_data.values(),
                    loader_memory.values(),
                )
            ]
        )

        try:
            for it in range(max_len):
                mb_curr_dict = {}
                mb_curr = []
                ReplayDataLoader._get_mini_batch_from_data_dict(
                    iter_data_dataloaders,
                    sampler_data,
                    loader_data,
                    self.oversample_small_tasks,
                    mb_curr,
                )
                mb_curr_dict['data'] = copy.deepcopy(mb_curr)
                ReplayDataLoader._get_mini_batch_from_data_dict(
                    iter_buffer_dataloaders,
                    sampler_memory,
                    loader_memory,
                    self.oversample_small_tasks,
                    mb_curr,
                )
                mb_curr_dict['mem'] = mb_curr[len(mb_curr_dict['data']):]
                yield self.collate_fn(mb_curr_dict)
        except StopIteration:
            return

def custom_collate_fn(mbatches):
    batch_dict = dict()
    batch_dict['data'] = torch.utils.data.default_collate(mbatches['data'])
    batch_dict['mem'] = torch.utils.data.default_collate(mbatches['mem'])
    return batch_dict

class ReplayPlugin(SupervisedPlugin):
    """
    Experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implementing `before_training_exp` and `after_training_exp`
    callbacks.
    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced
    such that there are the same number of examples for each experience.

    The `after_training_exp` callback is implemented in order to add new
    patterns to the external memory.

    The :mem_size: attribute controls the total number of patterns to be stored
    in the external memory.

    :param batch_size: the size of the data batch. If set to `None`, it
        will be set equal to the strategy's batch size.
    :param batch_size_mem: the size of the memory batch. If
        `task_balanced_dataloader` is set to True, it must be greater than or
        equal to the number of tasks. If its value is set to `None`
        (the default value), it will be automatically set equal to the
        data batch size.
    :param task_balanced_dataloader: if True, buffer data loaders will be
            task-balanced, otherwise it will create a single dataloader for the
            buffer samples.
    :param storage_policy: The policy that controls how to add new exemplars
                           in memory
    """

    def __init__(
        self,
        mem_size: int = 200,
        batch_size: int = None,
        batch_size_mem: int = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
    ):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader
        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy =CustomBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

    @property
    def ext_mem(self):
        return self.storage_policy.buffer_groups  # a Dict<task_id, Dataset>

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size
        if strategy.experience.current_experience != 0:
            ignored_params = list(map(id, strategy.model.conv1.parameters()))
            #ignored_params.extend(list(map(id, model.conv2.parameters())))
            base_params = filter(
                lambda p : id(p) not in ignored_params, strategy.model.parameters()
            )
            model_params = [
                {"params": base_params, "lr":0.001},
                #{"params":strategy.model.conv1.parameters(),
                #    "lr": 0,
                #},
            ]

            strategy.optimizer = torch.optim.Adam(model_params, lr=0.001)

        strategy.dataloader = CustomDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn = custom_collate_fn
        )
        return
    #def before_training_iteration(self, strategy: "SupervisedTemplate", **kwargs):
    def after_training_epoch(self, strategy: "SupervisedTemplate", **kwargs):
        del strategy.mbatch
        del strategy.loss
        del strategy.mb_output
        return
    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        feature_buffer = None
        targets = None
        task_ids = None
        for mbatch in strategy.dataloader:
            if isinstance(mbatch, dict):
            #if len(self.storage_policy.buffer) != 0:
                data = mbatch['data']
            else:
                data = mbatch
            with torch.no_grad():
                feature = strategy.model.get_features(data[0].cuda())
            if feature_buffer is None:
                feature_buffer = feature
                targets = data[1]
                task_ids = data[2]
            else:
                feature_buffer = torch.cat([feature_buffer, feature], 0)
                targets = torch.cat([targets, data[1]], 0)
                task_ids = torch.cat([task_ids, data[2]], 0)
        feature_buffer = feature_buffer.detach().cpu()
        targets = targets.detach().cpu()
        task_ids = task_ids.detach().cpu()
        #custom_buffer = AvalancheDataset(Buffer(feature_buffer, targets, task_ids))
        custom_buffer = MakeTensor(feature_buffer, targets, task_ids)
        self.storage_policy.update(strategy, custom_buffer, **kwargs)
        del feature_buffer
        del targets
        del task_ids
        gc.collect()
        return
