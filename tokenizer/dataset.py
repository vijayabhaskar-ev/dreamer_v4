
import abc
from typing import Iterator, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset
from .config import TokenizerConfig
from device_utils import get_ordinal

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
            seed = np.random.randint(0, 2**32 - 1)
        else:
            seed = worker_info.seed % (2**32 - 1)

        # Offset seed per TPU device so each chip gets different data
        seed = (seed + get_ordinal() * 17) % (2**32 - 1)
        np.random.seed(seed)
        env = self._get_env(seed=seed)
        
        for _ in range(self.steps_per_epoch): #TODO Need to refactor this after debugging the initial training pipleline.
            batch_videos = []
            batch_actions = []
            batch_rewards = []
            batch_dones = []
            for _ in range(self.batch_size):
                video = []
                actions = []
                raw_rewards = []
                raw_dones = []
                time_step = env.reset()

                # Sinusoidal policy: coordinated rhythmic joint movements
                # Randomize frequency and amplitude per episode for diversity
                action_dim = env.action_spec().shape[0]
                freq = np.random.uniform(0.5, 3.0)          # oscillation speed
                amplitude = np.random.uniform(0.3, 1.0)     # movement magnitude
                phase_offsets = np.array([i * np.pi / action_dim for i in range(action_dim)])

                for frame_idx in range(self.seq_len):
                    pixels = env.physics.render(
                        height=self.img_size[0],
                        width=self.img_size[1],
                        camera_id=self.camera_id
                    )

                    frame = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255.0
                    video.append(frame)

                    # Sinusoidal action: each joint follows a sine wave at different phase
                    t_norm = frame_idx / max(self.seq_len - 1, 1)
                    action = amplitude * np.sin(
                        t_norm * 2 * np.pi * freq + phase_offsets
                    )
                    # Clip to valid action range
                    action = np.clip(
                        action,
                        env.action_spec().minimum,
                        env.action_spec().maximum,
                    )

                    actions.append(torch.from_numpy(action.copy()).float())

                    step_reward = 0.0
                    step_done = False
                    for _ in range(self.action_repeat):
                        time_step = env.step(action)
                        step_reward += time_step.reward
                        if time_step.last():
                            step_done = True
                            break

                    raw_rewards.append(step_reward)
                    raw_dones.append(float(step_done))

                    if time_step.last():
                         # If episode ends early, just reset and continue filling this sequence.
                         # This creates a cut in the video, but for training it's acceptable.
                         time_step = env.reset()

                video_tensor = torch.stack(video)
                action_tensor = torch.stack(actions[:-1])
                reward_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
                done_tensor = torch.tensor(raw_dones, dtype=torch.float32)
                batch_videos.append(video_tensor)
                batch_actions.append(action_tensor)
                batch_rewards.append(reward_tensor)
                batch_dones.append(done_tensor)

            yield (torch.stack(batch_videos),   # (B, T, C, H, W)
                   torch.stack(batch_actions),  # (B, T-1, action_dim)
                   torch.stack(batch_rewards),  # (B, T)
                   torch.stack(batch_dones))    # (B, T)

class OfflineDataset(VideoDataset):
    """Load pre-generated episodes from a .npz file.

    Expected keys in the .npz:
        frames:  (N, T_total, H, W, 3)  uint8 0-255
        actions: (N, T_total, action_dim) float32
        rewards: (N, T_total)            float32  (optional, defaults to zeros)
        dones:   (N, T_total)            float32  (optional, defaults to zeros)

    Each iteration randomly picks an episode and a contiguous window
    of length ``seq_len``, returning (B, seq_len, C, H, W) frames,
    (B, seq_len-1, action_dim) actions, (B, seq_len) rewards,
    and (B, seq_len) dones.
    """

    def __init__(
        self,
        path: str,
        seq_len: int = 50,
        batch_size: int = 16,
        steps_per_epoch: int = 1000,
    ):
        data = np.load(path)
        self.frames = data["frames"]    # (N, T, H, W, 3)
        self.actions = data["actions"]  # (N, T, action_dim)
        T_total = self.frames.shape[1]
        N = self.frames.shape[0]
        self.rewards = data["rewards"] if "rewards" in data else np.zeros((N, T_total), dtype=np.float32)
        self.dones = data["dones"] if "dones" in data else np.zeros((N, T_total), dtype=np.float32)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.num_episodes = N
        self.episode_len = T_total

    def __iter__(self) -> Iterator[torch.Tensor]:
        # Offset seed per TPU device so each chip gets different data
        seed = (np.random.randint(0, 2**32 - 1) + get_ordinal() * 17) % (2**32 - 1)
        np.random.seed(seed)
        for _ in range(self.steps_per_epoch):
            idxs = np.random.randint(0, self.num_episodes, size=self.batch_size)
            max_start = max(0, self.episode_len - self.seq_len)
            starts = np.random.randint(0, max_start + 1, size=self.batch_size)

            batch_frames = []
            batch_actions = []
            batch_rewards = []
            batch_dones = []
            for idx, start in zip(idxs, starts):
                end = start + self.seq_len
                f = self.frames[idx, start:end]           # (seq_len, H, W, 3)
                a = self.actions[idx, start:end - 1]      # (seq_len-1, action_dim)
                r = self.rewards[idx, start:end]           # (seq_len,)
                d = self.dones[idx, start:end]             # (seq_len,)

                f = torch.from_numpy(f.copy()).permute(0, 3, 1, 2).float() / 255.0
                a = torch.from_numpy(a.copy()).float()
                r = torch.from_numpy(r.copy()).float()
                d = torch.from_numpy(d.copy()).float()
                batch_frames.append(f)
                batch_actions.append(a)
                batch_rewards.append(r)
                batch_dones.append(d)

            yield (
                torch.stack(batch_frames),   # (B, seq_len, C, H, W)
                torch.stack(batch_actions),  # (B, seq_len-1, action_dim)
                torch.stack(batch_rewards),  # (B, seq_len)
                torch.stack(batch_dones),    # (B, seq_len)
            )


class DatasetFactory:
    @staticmethod
    def get_dataset(
        config: TokenizerConfig,
        batch_size: int,
        steps_per_epoch: int = 1000,
        dataset_path: Optional[str] = None,
    ) -> IterableDataset:
        if config.dataset_name == "offline":
            if dataset_path is None:
                raise ValueError("--dataset-path is required when --dataset=offline")
            return OfflineDataset(
                path=dataset_path,
                seq_len=config.seq_len,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
            )
        elif config.dataset_name == "dm_control":
            return DMControlDataset(
                task_name=config.task_name,
                action_repeat=config.action_repeat,
                img_size=config.image_size,
                seq_len=config.seq_len,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch
            )
        elif config.dataset_name == "moving_square":
            from .train_tokenizer import MovingSquareDataset, MovingSquareIterableDataset
            ds = MovingSquareDataset(
                H=config.image_size[0],
                W=config.image_size[1],
                T=config.seq_len if hasattr(config, 'seq_len') else 4
            )
            return MovingSquareIterableDataset(ds, batch_size, steps_per_epoch=1000)

        else:
            raise ValueError(f"Unknown dataset: {config.dataset_name}")
