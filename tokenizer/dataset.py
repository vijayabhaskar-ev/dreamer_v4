
import abc
from typing import Iterator, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset
from .config import TokenizerConfig

class VideoDataset(IterableDataset, abc.ABC):
    """Abstract base class for video datasets."""
    
    @abc.abstractmethod
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yields batches of video sequences: (B, T, C, H, W)"""
        pass

class DMControlDataset(VideoDataset):
    """
    Dataset for DeepMind Control Suite tasks.
    Generates data on-the-fly by stepping through the environment.
    """
    def __init__(
        self, 
        task_name: str, 
        action_repeat: int = 2,
        img_size: Tuple[int, int] = (64, 64),
        seq_len: int = 16,
        batch_size: int = 16,
        camera_id: int = 0,
        steps_per_epoch: int = 1000,
    ):
        self.domain, self.task = task_name.split('_', 1)
        self.action_repeat = action_repeat
        self.img_size = img_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.camera_id = camera_id
        self.steps_per_epoch = steps_per_epoch
        
    def _get_env(self, seed: Optional[int] = None):
        from dm_control import suite
        task_kwargs = {}
        if seed is not None:
            task_kwargs['random'] = seed
        return suite.load(domain_name=self.domain, task_name=self.task, task_kwargs=task_kwargs)

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading, return the full iterator
            seed = np.random.randint(0, 2**32 - 1)
        else:
            # Multi-process data loading, split workload
            seed = worker_info.seed % (2**32 - 1)
        
        np.random.seed(seed)
        env = self._get_env(seed=seed)
        
        for _ in range(self.steps_per_epoch): #TODO Need to refactor this after debugging the initial training pipleline.
            batch_videos = []
            batch_actions= []
            for _ in range(self.batch_size):
                video = []
                actions = []
                time_step = env.reset()
                
                for _ in range(self.seq_len):
                    pixels = env.physics.render(
                        height=self.img_size[0], 
                        width=self.img_size[1], 
                        camera_id=self.camera_id
                    )

                    frame = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255.0
                    video.append(frame)

                    action = np.random.uniform(   #TODO Right now they mutiple workers may genarate same actions  as they copy from the parent process. Need to check this issue.
                        env.action_spec().minimum,
                        env.action_spec().maximum,
                        size=env.action_spec().shape
                    )

                    actions.append(torch.from_numpy(action.copy()).float())

                    print(f"Generated action: {action}")

                    for _ in range(self.action_repeat):
                        time_step = env.step(action)
                        if time_step.last():
                            break
                    
                    if time_step.last():
                         # If episode ends early, we could pad or just restart. 
                         # For simplicity, let's just reset and continue filling this sequence
                         # Note: This creates a cut in the video, but for tokenizer training it's acceptable
                         time_step = env.reset()

                video_tensor = torch.stack(video)
                action_tensor = torch.stack(actions[:-1])
                batch_videos.append(video_tensor)
                batch_actions.append(action_tensor)
            
            yield (torch.stack(batch_videos), # (B, T, C, H, W)
                  torch.stack(batch_actions)) # (B, T-1, action_dim)

class DatasetFactory:
    @staticmethod
    def get_dataset(config: TokenizerConfig, batch_size: int, steps_per_epoch: int = 1000) -> IterableDataset:
        if config.dataset_name == "dm_control":
            return DMControlDataset(
                task_name=config.task_name,
                action_repeat=config.action_repeat,
                img_size=config.image_size,
                seq_len=config.seq_len, # Assuming seq_len is in config, else default
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch
            )
        elif config.dataset_name == "moving_square":
            from .train_tokenizer import MovingSquareDataset, MovingSquareIterableDataset
            # Re-use existing logic (wrapped)
            ds = MovingSquareDataset(
                H=config.image_size[0], 
                W=config.image_size[1], 
                T=config.seq_len if hasattr(config, 'seq_len') else 4
            )
            # We need to wrap it to match the IterableDataset interface that yields batches
            # But MovingSquareIterableDataset logic is slightly different (yields one batch per step)
            # Let's assume the caller handles the wrapping or we standardize here.
            # For now, return the legacy wrapper if possible, or adapt.
            return MovingSquareIterableDataset(ds, batch_size, steps_per_epoch=1000) 
            
        else:
            raise ValueError(f"Unknown dataset: {config.dataset_name}")
