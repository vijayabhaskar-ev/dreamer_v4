# mock_data.py
import numpy as np
import torch

class MovingSquareDataset:
    """
    Creates synthetic video clips shaped (B, T, C, H, W).
    Perfect for debugging MAE tokenizer.
    """

    def __init__(self, H=64, W=64, T=4, C=3, square_size=8):
        self.H = H
        self.W = W
        self.T = T
        self.C = C
        self.square = square_size

    def sample(self, B):
        """
        Returns synthetic batch of videos. Shape:
        (B, T, C, H, W) with values in [0,1].
        """

        videos = np.zeros((B, self.T, self.C, self.H, self.W), dtype=np.float32)

        for b in range(B):
            x = np.random.randint(0, self.W - self.square)
            y = np.random.randint(0, self.H - self.square)

            dx = np.random.choice([-2, -1, 1, 2])
            dy = np.random.choice([-2, -1, 1, 2])

            for t in range(self.T):
                x = (x + dx) % (self.W - self.square)
                y = (y + dy) % (self.H - self.square)

                videos[b, t, :, y:y+self.square, x:x+self.square] = np.random.uniform(0.7, 1.0)

        return torch.from_numpy(videos)
