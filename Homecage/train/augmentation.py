import torch

class PoseAugmentation:
    def __init__(self, config=None, current_epoch=0):
        config = config or {}
        self.aug_schedule = config.get('augmentation', {}).get('aug_schedule', [])
        self.num_keypoints = config.get('model', {}).get('num_keypoints', 6)
        self.current_epoch = current_epoch

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_current_aug_params(self):
        for sched in self.aug_schedule:
            if ('start' in sched and 'end' in sched and sched['start'] <= self.current_epoch < sched['end']):
                return sched
        return self.aug_schedule[-1] if self.aug_schedule else {}

    def __call__(self, x):
        squeeze_batch = False
        if x.dim() == 4:
            x = x.unsqueeze(0)
            squeeze_batch = True
        if x.dim() != 5:
            raise ValueError(
                "node_position must be [T,N,K,2] or [B,T,N,K,2], "
                f"got {tuple(x.shape)}"
            )
        B, T, N, K, D = x.shape
        if D != 2:
            raise ValueError(
                "PoseAugmentation accepts position coordinates only; "
                f"expected D=2, got D={D}"
            )
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(
                "node_position contains NaN/Inf before augmentation; "
                "validity handling must happen in the data loader"
            )
        pos = x.clone()
        aug_params = self.get_current_aug_params()
        scale_range = float(aug_params.get('scale_range', 0.0))
        if scale_range > 0.0:
            scale = 1.0 + (torch.rand(1, device=pos.device).item() * 2.0 - 1.0) * scale_range
            center = pos.mean(dim=(2, 3), keepdim=True)
            pos = (pos - center) * scale + center
        noise_std = float(aug_params.get('noise_std', 0.0))
        if noise_std > 0.0:
            pos = pos + torch.randn_like(pos) * noise_std
        x = pos
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError("Pose augmentation produced NaN/Inf")
        if x.shape[2] != N or x.shape[3] != self.num_keypoints:
            raise ValueError(
                f"Augmentation error: {x.shape}, expected keypoint dim K={self.num_keypoints}"
            )
        if squeeze_batch:
            x = x.squeeze(0)
        return x
