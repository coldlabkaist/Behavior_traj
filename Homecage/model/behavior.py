import math

import torch
import torch.nn.functional as F


BEHAVIOR_TARGET_FEATURE_NAMES = [
    "social_spacing",
    "social_distance_delta",
    "social_isolation",
    "anogenital_investigation",
    "head_head_contact",
    "speed_rms",
    "accel_rms",
    "speed_p90",
    "speed_top20_mean",
    "speed_tail_spread",
    "accel_p90",
    "dominant_animal_ratio",
    "animal_speed_std",
    "activity_imbalance",
    "speed_synchrony",
]

SELECTED_BEHAVIOR_TARGET_FEATURE_NAMES = [
    "social_spacing",
    "social_isolation",
    "anogenital_investigation",
    "head_head_contact",
    "speed_rms",
    "speed_tail_spread",
    "dominant_animal_ratio",
]

ACTIVITY_DISTRIBUTION_FEATURE_NAMES = {
    "speed_p90",
    "speed_top20_mean",
    "speed_tail_spread",
    "accel_p90",
    "dominant_animal_ratio",
    "animal_speed_std",
}

DYNAMIC_PROFILE_FEATURE_NAMES = [
    "speed_top20_mean",
    "dominant_animal_ratio",
]


def normalize_behavior_feature_names(names) -> list:
    if names is None:
        return list(BEHAVIOR_TARGET_FEATURE_NAMES)
    if isinstance(names, str):
        names = [part.strip() for part in names.split(",")]
    out = []
    for name in names:
        text = str(name).strip()
        if not text:
            continue
        if text.startswith("beh::") or text.startswith("act::"):
            text = text.split("::", 1)[1]
        out.append(text.lower().replace(" ", "_").replace("-", "_"))
    return out or list(BEHAVIOR_TARGET_FEATURE_NAMES)


def _validate_input_tensor(x: torch.Tensor, expected_ndim: int = 5,
                           expected_last_dim: int = 2) -> tuple:
    if x is None:
        raise ValueError("Input tensor cannot be None")
    
    if x.ndim != expected_ndim:
        raise ValueError(
            f"Expected {expected_ndim}D tensor, got {x.ndim}D"
        )
    
    if expected_last_dim > 0 and x.size(-1) != expected_last_dim:
        raise ValueError(
            f"Expected last dimension to be {expected_last_dim}, "
            f"got {x.size(-1)}"
        )
    
    return tuple(x.shape)


def _clean_tensor(x: torch.Tensor, eps: float = 1e-8,
                   clamp_small_to_zero: bool = False) -> torch.Tensor:
    x_clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    if clamp_small_to_zero:
        x_clean = torch.where(
            torch.abs(x_clean) < eps,
            torch.zeros_like(x_clean),
            x_clean
        )
    
    return x_clean


def resolve_keypoint_index(keypoint_names, candidates, default=None):
    if keypoint_names is None:
        return default
    if isinstance(candidates, str):
        candidates = [candidates]
    names = [str(name).strip().lower() for name in keypoint_names]
    for candidate in candidates or []:
        cand = str(candidate).strip().lower()
        for idx, name in enumerate(names):
            if name == cand:
                return int(idx)
        for idx, name in enumerate(names):
            if cand in name or name in cand:
                return int(idx)
    return default

def compute_speed(
    x: torch.Tensor,
    *,
    fps: float = 30.0,
    mm_per_px: float = None,
    keypoint_idx: int = None,
    body_center_idx: int = None,
    smooth_window: int = 1,
    per_keypoint: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    B, T, N, K, _ = _validate_input_tensor(x, expected_ndim=5, expected_last_dim=2)
    if T < 2:
        raise ValueError(f"T must be >= 2 for speed calculation, got T={T}")
    
    x = _clean_tensor(x, eps=eps)
    
    if not per_keypoint:
        use_idx = keypoint_idx if keypoint_idx is not None else body_center_idx
        if use_idx is None:
            raise ValueError("keypoint_idx or body_center_idx must be provided for speed computation.")
        if use_idx < 0 or use_idx >= K:
            raise ValueError(f"Invalid keypoint index {use_idx}, must be in [0, {K-1}]")

    if per_keypoint:
        disp = x[:, 1:] - x[:, :-1]
        speed = torch.linalg.vector_norm(disp, dim=-1)
        pad = torch.zeros(B, 1, N, K, device=x.device, dtype=x.dtype)
        speed = torch.cat([pad, speed], dim=1)
    else:
        use_idx = keypoint_idx if keypoint_idx is not None else body_center_idx
        if use_idx is None:
            raise ValueError("keypoint_idx or body_center_idx must be provided for per-instance speed computation.")
        centers = x[:, :, :, use_idx, :]
        disp = centers[:, 1:] - centers[:, :-1]
        speed = torch.linalg.vector_norm(disp, dim=-1)
        pad = torch.zeros(B, 1, N, device=x.device, dtype=x.dtype)
        speed = torch.cat([pad, speed], dim=1)

    if fps and fps > 0:
        speed = speed * fps
    if mm_per_px is not None and mm_per_px > 0:
        speed = speed * mm_per_px
    
    speed = torch.clamp(speed, min=0.0)

    if isinstance(smooth_window, int) and smooth_window > 1:
        w = max(1, int(smooth_window))
        if per_keypoint:
            seq = speed.reshape(B * N * K, T)
        else:
            seq = speed.reshape(B * N, T)
        kernel = torch.ones(1, 1, w, device=x.device, dtype=x.dtype) / float(w)
        seq = seq.unsqueeze(1)  # [BN(K), 1, T]
        seq = torch.nn.functional.pad(seq, (w // 2, w - 1 - w // 2), mode="replicate")
        seq = torch.nn.functional.conv1d(seq, kernel)  # centered MA
        seq = seq.squeeze(1)
        speed = seq.reshape_as(speed)

    return speed

def compute_acceleration(
    x: torch.Tensor,
    *,
    fps: float = 30.0,
    mm_per_px: float = None,
    keypoint_idx: int = None,
    body_center_idx: int = None,
    smooth_window: int = 1,
    per_keypoint: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    B, T, N, K, _ = _validate_input_tensor(x, expected_ndim=5, expected_last_dim=2)

    if T < 3:
        raise ValueError(f"T must be >= 3 for acceleration calculation, got T={T}")

    x = _clean_tensor(x, eps=eps)
    
    use_idx = keypoint_idx if keypoint_idx is not None else body_center_idx
    if use_idx is None:
        raise ValueError("keypoint_idx or body_center_idx must be provided for acceleration computation.")
    if use_idx < 0 or use_idx >= K:
        raise ValueError(f"Invalid keypoint index {use_idx}, must be in [0, {K-1}]")

    if per_keypoint:
        p0 = x[:, :-2]
        p1 = x[:, 1:-1]
        p2 = x[:, 2:]
        acc = torch.linalg.vector_norm(p2 - 2 * p1 + p0, dim=-1)
        pad = torch.zeros(B, 2, N, K, device=x.device, dtype=x.dtype)
        acc = torch.cat([pad, acc], dim=1)
    else:
        use_idx = keypoint_idx if keypoint_idx is not None else body_center_idx
        if use_idx is None:
            raise ValueError("keypoint_idx or body_center_idx must be provided for per-instance acceleration computation.")
        centers = x[:, :, :, use_idx, :]
        p0 = centers[:, :-2]
        p1 = centers[:, 1:-1]
        p2 = centers[:, 2:]
        acc = torch.linalg.vector_norm(p2 - 2 * p1 + p0, dim=-1)
        pad = torch.zeros(B, 2, N, device=x.device, dtype=x.dtype)
        acc = torch.cat([pad, acc], dim=1)
    if fps and fps > 0:
        acc = acc * (fps ** 2)
    if mm_per_px is not None and mm_per_px > 0:
        acc = acc * mm_per_px

    acc = torch.clamp(acc, min=0.0)

    if isinstance(smooth_window, int) and smooth_window > 1:
        w = max(1, int(smooth_window))
        if per_keypoint:
            seq = acc.reshape(B * N * K, T)
        else:
            seq = acc.reshape(B * N, T)
        kernel = torch.ones(1, 1, w, device=x.device, dtype=x.dtype) / float(w)
        seq = seq.unsqueeze(1)
        seq = torch.nn.functional.pad(seq, (w // 2, w - 1 - w // 2), mode="replicate")
        seq = torch.nn.functional.conv1d(seq, kernel)
        seq = seq.squeeze(1)
        acc = seq.reshape_as(acc)

    return acc

def _angle_wrap(delta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(delta), torch.cos(delta))

def compute_angular_speed(
    x: torch.Tensor,
    *,
    fps: float = 30.0,
    keypoint_idx: int = None,
    body_center_idx: int = None,
    smooth_window: int = 1,
    pad_value: str = "zero",
) -> torch.Tensor:
    if x is None or x.ndim != 5 or x.size(-1) != 2:
        raise ValueError(f"x must be [B,T,N,K,2], got {tuple(x.shape) if isinstance(x, torch.Tensor) else type(x)}")
    B, T, N, K, _ = x.shape
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if T < 5:
        if pad_value == "nan":
            return torch.full((B, T, N), float('nan'), 
                            device=x.device, dtype=x.dtype)
        else:
            return torch.zeros(B, T, N, device=x.device, dtype=x.dtype)

    use_idx = keypoint_idx if keypoint_idx is not None else body_center_idx
    if use_idx is None:
        raise ValueError("keypoint_idx or body_center_idx must be provided for angular speed computation.")

    centers = x[:, :, :, use_idx, :]

    v_tilde = centers[:, 2:] - centers[:, :-2]
    theta = torch.atan2(v_tilde[..., 1], v_tilde[..., 0])

    dtheta = _angle_wrap(theta[:, 2:] - theta[:, :-2])
    dt = 1.0 / fps if fps and fps > 0 else 1.0
    omega_mid = dtheta / (2.0 * dt)

    omega = torch.zeros(B, T, N, device=x.device, dtype=x.dtype)
    omega[:, 2:-2, :] = omega_mid

    if isinstance(smooth_window, int) and smooth_window > 1:
        w = max(1, int(smooth_window))
        seq = omega.reshape(B * N, T)
        kernel = torch.ones(1, 1, w, device=x.device, dtype=x.dtype) / float(w)
        seq = seq.unsqueeze(1)
        seq = F.pad(seq, (w // 2, w - 1 - w // 2), mode="replicate")
        seq = F.conv1d(seq, kernel)
        omega = seq.squeeze(1).reshape_as(omega)

    return omega

def compute_relative_head_angle(
    x: torch.Tensor,
    *,
    nose_idx: int,
    neck_idx: int,
    body_center_idx: int,
    smooth_window: int = 1,
    invert_y: bool = False,
) -> torch.Tensor:
    if x.ndim != 5 or x.size(-1) != 2:
        raise ValueError("x must be [B,T,N,K,2]")
    B, T, N, K, _ = x.shape
    nose = x[:, :, :, nose_idx, :]
    neck = x[:, :, :, neck_idx, :]
    body = x[:, :, :, body_center_idx, :]
    if invert_y:
        nose = nose * torch.tensor([1.0, -1.0], device=x.device, dtype=x.dtype)
        neck = neck * torch.tensor([1.0, -1.0], device=x.device, dtype=x.dtype)
        body = body * torch.tensor([1.0, -1.0], device=x.device, dtype=x.dtype)
    h = nose - neck
    u = neck - body
    theta_h = torch.atan2(h[..., 1], h[..., 0])
    theta_u = torch.atan2(u[..., 1], u[..., 0])
    phi = _angle_wrap(theta_h - theta_u)

    if isinstance(smooth_window, int) and smooth_window > 1:
        w = int(smooth_window)
        seq = phi.reshape(B * N, T)
        kernel = torch.ones(1, 1, w, device=x.device, dtype=x.dtype) / float(w)
        seq = seq.unsqueeze(1)
        seq = F.pad(seq, (w // 2, w - 1 - w // 2), mode="replicate")
        seq = F.conv1d(seq, kernel).squeeze(1)
        phi = seq.reshape(B, T, N)
    return phi

def compute_inter_individual_distance(
    x: torch.Tensor,
    *,
    mask: torch.Tensor = None,
    body_center_idx: int = None,
    keypoint_idx: int = None,
    mm_per_px: float = None,
) -> torch.Tensor:
    B, T, N, K, _ = _validate_input_tensor(x, expected_ndim=5, expected_last_dim=2)
    
    use_idx = keypoint_idx if keypoint_idx is not None else body_center_idx
    if use_idx is None:
        raise ValueError("keypoint_idx or body_center_idx must be provided.")
    
    if use_idx < 0 or use_idx >= K:
        raise ValueError(f"Invalid keypoint index {use_idx}, must be in [0, {K-1}]")

    P = _clean_tensor(x[..., use_idx, :], eps=1e-8)
    
    if mm_per_px is not None:
        if mm_per_px <= 0:
            mm_per_px = None
        else:
            P = P * mm_per_px

    PT = P.reshape(B*T, N, 2)
    D = torch.cdist(PT, PT, p=2).reshape(B, T, N, N)

    if mask is not None:
        m = mask[..., use_idx].bool()
        pair_valid = m.unsqueeze(-1) & m.unsqueeze(-2)
        D = torch.where(pair_valid, D, torch.tensor(float('nan'), device=D.device, dtype=D.dtype))

    eye = torch.eye(N, device=D.device, dtype=torch.bool).view(1,1,N,N)
    D = torch.where(eye, torch.zeros((), device=D.device, dtype=D.dtype), D)

    return D

def compute_distance_summary(
    D: torch.Tensor,
    *,
    summary_type: str = 'mean',
    exclude_self: bool = True,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    if D is None or D.ndim != 4:
        raise ValueError(f"D must be [B,T,N,N], got {tuple(D.shape) if isinstance(D, torch.Tensor) else type(D)}")
    
    B, T, N, _ = D.shape
    X = D

    if exclude_self:
        eye = torch.eye(N, device=D.device, dtype=torch.bool).view(1,1,N,N)
        X = X.masked_fill(eye, float('nan'))

    if mask is not None:
        if mask.shape != (B,T,N):
            raise ValueError(f"mask must be [B,T,N], got {mask.shape}")
        row_valid = mask.bool()
    else:
        row_valid = torch.ones(B,T,N, device=D.device, dtype=torch.bool)

    if summary_type == 'mean':
        out = torch.nanmean(X, dim=-1)
    elif summary_type == 'min':
        out = torch.nanmin(X, dim=-1).values
    elif summary_type == 'max':
        out = torch.nanmax(X, dim=-1).values
    elif summary_type == 'median':
        if hasattr(torch, 'nanmedian'):
            out = torch.nanmedian(X, dim=-1).values
        else:
            out = torch.nanquantile(X, 0.5, dim=-1)
    else:
        raise ValueError(f"summary_type must be one of ['mean','min','max','median'], got {summary_type}")

    out = torch.where(row_valid, out, torch.tensor(float('nan'), device=out.device, dtype=out.dtype))
    return out


def compute_relative_orientation(
    x: torch.Tensor,
    *,
    mask: torch.Tensor = None,
    nose_idx: int,
    neck_idx: int,
    body_center_idx: int,
    heading_source: str = "head",
    smooth_window: int = 1,
    return_angle: bool = False,
    eps: float = 1e-6,
    invert_y: bool = False,
):

    B, T, N, K, _ = _validate_input_tensor(x, expected_ndim=5, expected_last_dim=2)
    
    if nose_idx < 0 or nose_idx >= K:
        raise ValueError(f"Invalid nose_idx {nose_idx}, must be in [0, {K-1}]")
    if neck_idx < 0 or neck_idx >= K:
        raise ValueError(f"Invalid neck_idx {neck_idx}, must be in [0, {K-1}]")
    if body_center_idx < 0 or body_center_idx >= K:
        raise ValueError(f"Invalid body_center_idx {body_center_idx}, must be in [0, {K-1}]")
    
    X = _clean_tensor(x, eps=eps)
    if invert_y:
        Yflip = torch.tensor([1.0, -1.0], device=X.device, dtype=X.dtype)
        X = X * Yflip

    if heading_source == "head":
        h = X[..., nose_idx, :] - X[..., neck_idx, :]
        heading_valid = None if mask is None else (mask[..., nose_idx] & mask[..., neck_idx])
    elif heading_source == "trunk":
        h = X[..., neck_idx, :] - X[..., body_center_idx, :]
        heading_valid = None if mask is None else (mask[..., neck_idx] & mask[..., body_center_idx])
    elif heading_source == "velocity":
        c = X[..., body_center_idx, :]
        h = torch.zeros_like(c)
        h[:,1:,:,:] = c[:,1:,:,:] - c[:,:-1,:,:]
        heading_valid = None if mask is None else (mask[..., body_center_idx] & torch.roll(mask[..., body_center_idx], 1, dims=1))
    else:
        raise ValueError("heading_source must be one of {'head','trunk','velocity'}")

    if smooth_window > 1:
        smooth_window = max(1, int(smooth_window))
        w = torch.ones(1,1,smooth_window, device=X.device, dtype=X.dtype) / float(smooth_window)
        hv = h.reshape(B*N, T, 2).permute(0,2,1)
        hv = F.pad(hv, (smooth_window//2, smooth_window-1-smooth_window//2), mode="replicate")
        hv = F.conv1d(hv, w.expand(2,1,-1), groups=2)
        h = hv.permute(0,2,1).reshape(B,T,N,2)

    h_norm = torch.linalg.vector_norm(h, dim=-1, keepdim=True)
    valid_heading = h_norm > eps
    u = torch.where(valid_heading, h / h_norm, torch.zeros_like(h))

    p = X[..., body_center_idx, :]
    pi = p.unsqueeze(3)
    pj = p.unsqueeze(2)
    b = pj - pi

    b_norm = torch.linalg.vector_norm(b, dim=-1, keepdim=True)
    valid_bearing = b_norm > eps
    b_hat = torch.where(valid_bearing, b / b_norm, torch.zeros_like(b))

    ui = u.unsqueeze(3)
    cos_phi = (ui * b_hat).sum(dim=-1)

    if return_angle:
        cross = ui[...,0]*b_hat[...,1] - ui[...,1]*b_hat[...,0]
        phi = torch.atan2(cross, cos_phi)
    else:
        phi = None

    if mask is not None:
        valid_i = mask[..., body_center_idx]
        valid_j = valid_i
        if heading_valid is not None:
            valid_i = valid_i & heading_valid
        pair_valid = valid_i.unsqueeze(-1) & valid_j.unsqueeze(-2)
    else:
        pair_valid = torch.ones(B,T,N,N, dtype=torch.bool, device=X.device)

    eye = torch.eye(N, device=X.device, dtype=torch.bool).view(1,1,N,N)
    pair_valid = pair_valid & (~eye) & (b_norm.squeeze(-1) > eps)

    nan = torch.tensor(float('nan'), device=X.device, dtype=X.dtype)
    cos_phi = torch.where(pair_valid, cos_phi, nan)
    if phi is not None:
        phi = torch.where(pair_valid, phi, nan)
    
    cos_phi = torch.clamp(cos_phi, -1.0, 1.0)

    return (cos_phi, phi) if return_angle else cos_phi


def _compute_heading_unit_vectors(
    x: torch.Tensor,
    *,
    mask: torch.Tensor = None,
    nose_idx: int,
    neck_idx: int,
    body_center_idx: int,
    heading_source: str = "head",
    smooth_window: int = 1,
    eps: float = 1e-6,
    invert_y: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, N, K, _ = _validate_input_tensor(x, expected_ndim=5, expected_last_dim=2)

    if nose_idx < 0 or nose_idx >= K:
        raise ValueError(f"Invalid nose_idx {nose_idx}, must be in [0, {K-1}]")
    if neck_idx < 0 or neck_idx >= K:
        raise ValueError(f"Invalid neck_idx {neck_idx}, must be in [0, {K-1}]")
    if body_center_idx < 0 or body_center_idx >= K:
        raise ValueError(f"Invalid body_center_idx {body_center_idx}, must be in [0, {K-1}]")

    X = _clean_tensor(x, eps=eps)
    if invert_y:
        Yflip = torch.tensor([1.0, -1.0], device=X.device, dtype=X.dtype)
        X = X * Yflip

    if heading_source == "head":
        h = X[..., nose_idx, :] - X[..., neck_idx, :]
        heading_valid = None if mask is None else (mask[..., nose_idx] & mask[..., neck_idx])
    elif heading_source == "trunk":
        h = X[..., neck_idx, :] - X[..., body_center_idx, :]
        heading_valid = None if mask is None else (mask[..., neck_idx] & mask[..., body_center_idx])
    elif heading_source == "velocity":
        c = X[..., body_center_idx, :]
        h = torch.zeros_like(c)
        h[:, 1:, :, :] = c[:, 1:, :, :] - c[:, :-1, :, :]
        heading_valid = None if mask is None else (mask[..., body_center_idx] & torch.roll(mask[..., body_center_idx], 1, dims=1))
    else:
        raise ValueError("heading_source must be one of {'head','trunk','velocity'}")

    if smooth_window > 1:
        smooth_window = max(1, int(smooth_window))
        w = torch.ones(1, 1, smooth_window, device=X.device, dtype=X.dtype) / float(smooth_window)
        hv = h.reshape(B * N, T, 2).permute(0, 2, 1)
        hv = F.pad(hv, (smooth_window // 2, smooth_window - 1 - smooth_window // 2), mode="replicate")
        hv = F.conv1d(hv, w.expand(2, 1, -1), groups=2)
        h = hv.permute(0, 2, 1).reshape(B, T, N, 2)

    h_norm = torch.linalg.vector_norm(h, dim=-1, keepdim=True)
    valid_heading = h_norm > eps
    u = torch.where(valid_heading, h / h_norm, torch.zeros_like(h))
    return X, u, heading_valid


def compute_heading_alignment(
    x: torch.Tensor,
    *,
    mask: torch.Tensor = None,
    nose_idx: int,
    neck_idx: int,
    body_center_idx: int,
    heading_source: str = "head",
    smooth_window: int = 1,
    eps: float = 1e-6,
    invert_y: bool = False,
) -> torch.Tensor:
    X, u, heading_valid = _compute_heading_unit_vectors(
        x,
        mask=mask,
        nose_idx=nose_idx,
        neck_idx=neck_idx,
        body_center_idx=body_center_idx,
        heading_source=heading_source,
        smooth_window=smooth_window,
        eps=eps,
        invert_y=invert_y,
    )
    B, T, N, _, _ = X.shape
    align = (u.unsqueeze(3) * u.unsqueeze(2)).sum(dim=-1)

    if mask is not None:
        valid = mask[..., body_center_idx]
        if heading_valid is not None:
            valid = valid & heading_valid
        pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    else:
        pair_valid = torch.ones((B, T, N, N), dtype=torch.bool, device=X.device)

    eye = torch.eye(N, device=X.device, dtype=torch.bool).view(1, 1, N, N)
    pair_valid = pair_valid & (~eye)
    nan = torch.tensor(float("nan"), device=X.device, dtype=X.dtype)
    align = torch.where(pair_valid, align, nan)
    return torch.clamp(align, -1.0, 1.0)


def compute_directed_attention_scores(
    x: torch.Tensor,
    *,
    mask: torch.Tensor = None,
    nose_idx: int,
    neck_idx: int,
    body_center_idx: int,
    heading_source: str = "head",
    smooth_window: int = 1,
    cone_angle_deg: float = 60.0,
    attention_temp: float = 0.15,
    distance_tau: float = 0.6,
    eps: float = 1e-6,
    invert_y: bool = False,
) -> torch.Tensor:
    B, T, N, K, _ = _validate_input_tensor(x, expected_ndim=5, expected_last_dim=2)
    if body_center_idx < 0 or body_center_idx >= K:
        raise ValueError(f"Invalid body_center_idx {body_center_idx}, must be in [0, {K-1}]")

    cos_phi = compute_relative_orientation(
        x,
        mask=mask,
        nose_idx=nose_idx,
        neck_idx=neck_idx,
        body_center_idx=body_center_idx,
        heading_source=heading_source,
        smooth_window=smooth_window,
        eps=eps,
        invert_y=invert_y,
    )

    X = _clean_tensor(x, eps=eps)
    if invert_y:
        Yflip = torch.tensor([1.0, -1.0], device=X.device, dtype=X.dtype)
        X = X * Yflip

    centers = X[..., body_center_idx, :]
    dist = torch.cdist(centers.reshape(B * T, N, 2), centers.reshape(B * T, N, 2), p=2).reshape(B, T, N, N)
    pair_valid = torch.isfinite(cos_phi) & torch.isfinite(dist)

    cos_thresh = float(math.cos(math.radians(float(cone_angle_deg))))
    temperature = float(max(attention_temp, eps))
    angle_term = torch.sigmoid((torch.nan_to_num(cos_phi, nan=-1.0) - cos_thresh) / temperature)

    if distance_tau is not None and float(distance_tau) > 0.0:
        tau = float(max(distance_tau, eps))
        dist_term = torch.exp(-(dist * dist) / (tau * tau))
    else:
        dist_term = torch.ones_like(angle_term)

    attn = angle_term * dist_term
    eye = torch.eye(N, device=attn.device, dtype=torch.bool).view(1, 1, N, N)
    attn = torch.where(pair_valid & (~eye), attn, torch.zeros_like(attn))
    return torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)


def compute_pairwise_synchrony_corr(
    signal: torch.Tensor,
    *,
    mask: torch.Tensor = None,
    half_window: int = 15,
    min_count: int = 5,
    variance_threshold: float = 1e-8,
    eps: float = 1e-8,
) -> torch.Tensor:
    if signal.ndim != 3:
        raise ValueError(f"signal must be [B,T,N], got {signal.shape}")
    B, T, N = signal.shape
    w = half_window
    W = 2*w + 1
    x = torch.nan_to_num(signal, nan=0.0).to(dtype=signal.dtype)

    if mask is None:
        m = torch.ones(B, T, N, dtype=torch.bool, device=x.device)
    else:
        if mask.shape != (B, T, N):
            raise ValueError(f"mask must be [B,T,N], got {mask.shape}")
        m = mask.bool()

    xi = x.unsqueeze(3)
    xj = x.unsqueeze(2)
    mi = m.unsqueeze(3).to(dtype=x.dtype)
    mj = m.unsqueeze(2).to(dtype=x.dtype)
    pij = (mi * mj)

    kernel = torch.ones(1, 1, W, device=x.device, dtype=x.dtype)
    nan_tensor = torch.tensor(float('nan'), device=x.device, dtype=x.dtype)

    def wsum(z):
        z = z.permute(0, 2, 3, 1).reshape(B*N*N, 1, T)
        z = F.conv1d(z, kernel, padding=w)
        return z.reshape(B, N, N, T).permute(0, 3, 1, 2)

    sum_m = wsum(pij)
    sum_xi = wsum(xi * pij)
    sum_xj = wsum(xj * pij)
    sum_xixi = wsum((xi*xi) * pij)
    sum_xjxj = wsum((xj*xj) * pij)
    sum_xixj = wsum((xi*xj) * pij)

    num = sum_xixj - (sum_xi * sum_xj) / (sum_m.clamp_min(1.0))
    var_i = sum_xixi - (sum_xi**2) / (sum_m.clamp_min(1.0))
    var_j = sum_xjxj - (sum_xj**2) / (sum_m.clamp_min(1.0))
    den = (var_i.clamp_min(0.0) * var_j.clamp_min(0.0)).sqrt().clamp_min(eps)
    R = (num / den).clamp(-1.0, 1.0)

    eye = torch.eye(N, device=x.device, dtype=torch.bool).view(1, 1, N, N)
    valid = (~eye) & (sum_m >= float(min_count)) & (var_i >= variance_threshold) & (var_j >= variance_threshold)
    R = torch.where(valid, R, nan_tensor)
    return R


def compute_speed_from_coords(
    x: torch.Tensor,
    *, 
    body_center_idx: int,
    fps: float,
    mm_per_px: float = None,
) -> torch.Tensor:
    if x.ndim != 5 or x.size(-1) != 2:
        raise ValueError("x must be [B,T,N,K,2]")
    c = torch.nan_to_num(x[..., body_center_idx, :], nan=0.0)
    v = torch.zeros_like(c)
    if fps and fps > 0:
        v[:, 1:, :, :] = (c[:, 1:, :, :] - c[:, :-1, :, :]) * fps
    else:
        v[:, 1:, :, :] = (c[:, 1:, :, :] - c[:, :-1, :, :])
    speed = torch.linalg.vector_norm(v, dim=-1)
    if mm_per_px is not None and mm_per_px > 0:
        speed = speed * mm_per_px
    return speed


def behavioral_synchrony_speed_corr(
    x: torch.Tensor,
    *,
    body_center_idx: int,
    fps: float,
    mm_per_px: float = None,
    mask: torch.Tensor = None,
    half_window: int = 15,
    use_delta: bool = True,
    min_count: int = 5,
) -> torch.Tensor:
    
    speed = compute_speed_from_coords(
        x, body_center_idx=body_center_idx, fps=fps, mm_per_px=mm_per_px
    )
    if use_delta:
        ds = torch.zeros_like(speed)
        ds[:, 1:, :] = speed[:, 1:, :] - speed[:, :-1, :]
        signal = ds
        
        if mask is not None:
            m = mask[..., body_center_idx]
            delta_mask = torch.zeros_like(m)
            delta_mask[:, 1:, :] = m[:, 1:, :] & m[:, :-1, :]
            m = delta_mask
        else:
            m = None
    else:
        signal = speed
        m = None if mask is None else mask[..., body_center_idx]
        
    return compute_pairwise_synchrony_corr(
        signal, mask=m, half_window=half_window, min_count=min_count
    )

def compute_group_inter_distance_mean(
    x: torch.Tensor,
    *,
    body_center_idx: int,
    mm_per_px: float = None,
    mask: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    D = compute_inter_individual_distance(
        x, mask=mask, body_center_idx=body_center_idx, mm_per_px=mm_per_px
    )
    return _compute_pairwise_mean(D, mask, body_center_idx, eps)


def _convex_hull_area_single(points_xy: torch.Tensor) -> torch.Tensor:
    if points_xy.ndim != 2 or points_xy.size(-1) != 2:
        raise ValueError(f"points_xy must be [N,2], got {tuple(points_xy.shape)}")

    valid = torch.isfinite(points_xy).all(dim=-1)
    pts = points_xy[valid]
    if pts.shape[0] < 3:
        return points_xy.new_tensor(0.0)

    pts_list = sorted({(float(p[0]), float(p[1])) for p in pts.detach().cpu()})
    if len(pts_list) < 3:
        return points_xy.new_tensor(0.0)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts_list:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts_list):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return points_xy.new_tensor(0.0)

    hull_xy = points_xy.new_tensor(hull)
    x = hull_xy[:, 0]
    y = hull_xy[:, 1]
    area = 0.5 * torch.abs(torch.sum(x * torch.roll(y, shifts=-1) - y * torch.roll(x, shifts=-1)))
    return area


def compute_group_convex_hull_area(
    x: torch.Tensor,
    *,
    body_center_idx: int,
    mm_per_px: float = None,
    mask: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    B, T, N, K, _ = _validate_input_tensor(x, expected_ndim=5, expected_last_dim=2)
    use_idx = int(body_center_idx)
    if use_idx < 0 or use_idx >= K:
        raise ValueError(f"Invalid body_center_idx {use_idx}, must be in [0, {K-1}]")

    x = _clean_tensor(x, eps=eps)
    centers = x[:, :, :, use_idx, :]  # [B, T, N, 2]

    # Fast path for the current setup: with 3 animals, the convex hull is
    # just the triangle area formed by the three body centers.
    if N == 3:
        p0 = centers[:, :, 0, :]
        p1 = centers[:, :, 1, :]
        p2 = centers[:, :, 2, :]
        tri_cross = (
            (p1[..., 0] - p0[..., 0]) * (p2[..., 1] - p0[..., 1])
            - (p1[..., 1] - p0[..., 1]) * (p2[..., 0] - p0[..., 0])
        )
        area = 0.5 * torch.abs(tri_cross)
        valid = torch.isfinite(centers).all(dim=-1).all(dim=-1)
        if mask is not None:
            if mask.shape[:4] != x.shape[:4]:
                raise ValueError(
                    f"mask must match x on first 4 dims, got mask={tuple(mask.shape)} x={tuple(x.shape)}"
                )
            center_mask = (mask[..., use_idx] > 0).all(dim=-1)
            valid = valid & center_mask
        area = torch.where(valid, area, torch.zeros_like(area))
        if mm_per_px is not None and mm_per_px > 0:
            area = area * float(mm_per_px) * float(mm_per_px)
        return torch.nan_to_num(area, nan=0.0, posinf=0.0, neginf=0.0)

    center_mask = None
    if mask is not None:
        if mask.shape[:4] != x.shape[:4]:
            raise ValueError(
                f"mask must match x on first 4 dims, got mask={tuple(mask.shape)} x={tuple(x.shape)}"
            )
        center_mask = mask[..., use_idx].reshape(B * T, N) > 0

    flat_centers = centers.reshape(B * T, N, 2)
    flat_area = centers.new_zeros(B * T)
    for idx in range(flat_centers.shape[0]):
        pts = flat_centers[idx]
        if center_mask is not None:
            pts = pts[center_mask[idx]]
        flat_area[idx] = _convex_hull_area_single(pts)

    area = flat_area.view(B, T)
    if mm_per_px is not None and mm_per_px > 0:
        area = area * float(mm_per_px) * float(mm_per_px)
    return torch.nan_to_num(area, nan=0.0, posinf=0.0, neginf=0.0)


def compute_group_synchronization(
    x: torch.Tensor,
    *,
    mask: torch.Tensor = None,
    body_center_idx: int,
    fps: float,
    mm_per_px: float = None,
    half_window: int = 15,
    use_delta: bool = True,
    min_count: int = 5,
    metric: str = "speed_correlation",
    events: torch.Tensor = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if metric == "speed_correlation":
        R = behavioral_synchrony_speed_corr(
            x,
            body_center_idx=body_center_idx,
            fps=fps,
            mm_per_px=mm_per_px,
            mask=mask,
            half_window=half_window,
            use_delta=use_delta,
            min_count=min_count,
        )
        return _compute_pairwise_mean(R, mask, body_center_idx, eps)
        
    else:
        raise ValueError(
            f"metric must be 'speed_correlation', got {metric}"
        )


def _compute_pairwise_mean(
    matrix: torch.Tensor,
    mask: torch.Tensor,
    keypoint_idx: int,
    eps: float,
) -> torch.Tensor:
    B, T, N, _ = matrix.shape
    
    # Use only unique inter-individual pairs (upper triangle, excluding diagonal).
    valid_pairs = torch.triu(
        torch.ones((1, 1, N, N), device=matrix.device, dtype=torch.bool),
        diagonal=1,
    )

    matrix_valid = torch.where(
        valid_pairs,
        matrix,
        torch.tensor(float('nan'), device=matrix.device),
    )
    
    pairwise_mean = torch.nanmean(matrix_valid, dim=(2, 3))
    
    return pairwise_mean


def _offdiag_mean(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 4:
        raise ValueError(f"matrix must be [B,T,N,N], got {tuple(matrix.shape)}")
    _, _, n, _ = matrix.shape
    valid = ~torch.eye(n, device=matrix.device, dtype=torch.bool).view(1, 1, n, n)
    masked = torch.where(valid, matrix, torch.tensor(float("nan"), device=matrix.device, dtype=matrix.dtype))
    return torch.nanmean(masked, dim=(2, 3))


def _soft01_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def _activity_gmm_features(
    *,
    speed_rms: torch.Tensor,
    accel_rms: torch.Tensor,
    activity_imbalance: torch.Tensor,
    params: dict,
    eps: float = 1e-8,
) -> dict:
    if not params:
        raise ValueError(
            "Activity-state target requested, but no activity_gmm_params were provided. "
            "Set training.behavior_activity_gmm_params_path or model.behavior_activity_gmm_params_path."
        )

    device = speed_rms.device
    dtype = speed_rms.dtype
    x = torch.stack(
        [
            torch.log1p(torch.clamp(speed_rms, min=0.0)),
            torch.log1p(torch.clamp(accel_rms, min=0.0)),
            activity_imbalance,
        ],
        dim=-1,
    )
    scaler_mean = torch.as_tensor(params.get("scaler_mean"), device=device, dtype=dtype)
    scaler_scale = torch.as_tensor(params.get("scaler_scale"), device=device, dtype=dtype).clamp_min(eps)
    weights = torch.as_tensor(params.get("gmm_weights_raw"), device=device, dtype=dtype).clamp_min(eps)
    means = torch.as_tensor(params.get("gmm_means_raw_order"), device=device, dtype=dtype)
    covs = torch.as_tensor(params.get("gmm_covariances_raw_order"), device=device, dtype=dtype)
    if scaler_mean.numel() != 3 or scaler_scale.numel() != 3 or means.ndim != 2 or covs.ndim != 3:
        raise ValueError("Invalid activity GMM parameters.")

    z = (x - scaler_mean.view(1, 1, -1)) / scaler_scale.view(1, 1, -1)
    n_comp = int(means.shape[0])
    dim = int(means.shape[1])
    eye = torch.eye(dim, device=device, dtype=dtype).view(1, dim, dim)
    covs = covs + eye * 1e-6
    inv_covs = torch.linalg.inv(covs)
    sign, logdet = torch.linalg.slogdet(covs)
    logdet = torch.where(sign > 0, logdet, torch.zeros_like(logdet))
    diff = z.unsqueeze(-2) - means.view(1, 1, n_comp, dim)
    maha = torch.einsum("...ci,cij,...cj->...c", diff, inv_covs, diff)
    const = float(dim) * math.log(2.0 * math.pi)
    log_prob = -0.5 * (maha + logdet.view(1, 1, n_comp) + const)
    log_prob = log_prob + torch.log(weights / weights.sum()).view(1, 1, n_comp)
    posterior = torch.softmax(log_prob, dim=-1)

    order = params.get("component_order_by_log_speed", None)
    if order is not None:
        order_t = torch.as_tensor(order, device=device, dtype=torch.long)
        posterior = posterior.index_select(dim=-1, index=order_t)

    p_inactive = posterior[..., 0]
    if posterior.shape[-1] == 1:
        p_active = torch.zeros_like(p_inactive)
        p_high = torch.zeros_like(p_inactive)
    elif posterior.shape[-1] == 2:
        p_active = posterior[..., 1]
        p_high = torch.zeros_like(p_active)
    else:
        p_active = posterior[..., 1]
        p_high = posterior[..., 2:].sum(dim=-1)
    p_any = torch.clamp(p_active + p_high, 0.0, 1.0)
    probs = posterior.clamp_min(eps)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1) / math.log(max(2, posterior.shape[-1]))
    return {
        "p_inactive": p_inactive,
        "p_active": p_active,
        "p_high_activity": p_high,
        "p_any_activity": p_any,
        "soft_activity_index": torch.clamp(p_active + 2.0 * p_high, 0.0, 2.0) / 2.0,
        "state_entropy": entropy,
        "active_intensity": p_any * speed_rms,
    }


def _quantile_1d(x: torch.Tensor, q: float, dim: int) -> torch.Tensor:
    return torch.quantile(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), q=float(q), dim=dim)


def _top_fraction_mean(x: torch.Tensor, fraction: float, dim: int) -> torch.Tensor:
    size = int(x.shape[dim])
    k = max(1, int(math.ceil(float(fraction) * float(size))))
    values = torch.topk(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), k=k, dim=dim).values
    return values.mean(dim=dim)


def _frame_activity_distribution_features(
    speed: torch.Tensor,
    accel: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict:
    speed_p50 = _quantile_1d(speed, 0.50, dim=-1)
    speed_p90 = _quantile_1d(speed, 0.90, dim=-1)
    accel_p90 = _quantile_1d(accel, 0.90, dim=-1)
    speed_sum = speed.sum(dim=-1)
    return {
        "speed_p90": speed_p90,
        "speed_top20_mean": _top_fraction_mean(speed, 0.20, dim=-1),
        "speed_tail_spread": speed_p90 - speed_p50,
        "accel_p90": accel_p90,
        "dominant_animal_ratio": speed.max(dim=-1).values / speed_sum.clamp_min(eps),
        "animal_speed_std": torch.std(speed, dim=-1, unbiased=False),
    }


def _activity_distribution_from_window(
    speed_w: torch.Tensor,
    accel_w: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    if speed_w.ndim != 3:
        raise ValueError(f"speed_w must be [B,L,N], got {tuple(speed_w.shape)}")
    b = int(speed_w.shape[0])
    speed_w = torch.nan_to_num(speed_w, nan=0.0, posinf=0.0, neginf=0.0)
    accel_w = torch.nan_to_num(accel_w, nan=0.0, posinf=0.0, neginf=0.0)
    speed_flat = speed_w.reshape(b, -1)
    accel_flat = accel_w.reshape(b, -1)
    speed_p50 = _quantile_1d(speed_flat, 0.50, dim=1)
    speed_p90 = _quantile_1d(speed_flat, 0.90, dim=1)
    animal_speed_sum = speed_w.sum(dim=1)
    animal_speed_mean = speed_w.mean(dim=1)
    total_speed = animal_speed_sum.sum(dim=-1)
    return {
        "speed_p90": speed_p90,
        "speed_top20_mean": _top_fraction_mean(speed_flat, 0.20, dim=1),
        "speed_tail_spread": speed_p90 - speed_p50,
        "accel_p90": _quantile_1d(accel_flat, 0.90, dim=1),
        "dominant_animal_ratio": animal_speed_sum.max(dim=-1).values / total_speed.clamp_min(eps),
        "animal_speed_std": torch.std(animal_speed_mean, dim=-1, unbiased=False),
    }


def _window_activity_distribution_features(
    speed: torch.Tensor,
    accel: torch.Tensor,
    *,
    window_size: int,
    stride: int,
    num_tokens: int,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    if speed.ndim != 3:
        raise ValueError(f"speed must be [B,T,N], got {tuple(speed.shape)}")
    b, t, _ = speed.shape
    window_size = max(1, int(window_size))
    stride = max(1, int(stride))
    num_tokens = max(1, int(num_tokens))
    rows: dict[str, list[torch.Tensor]] = {name: [] for name in ACTIVITY_DISTRIBUTION_FEATURE_NAMES}

    for i in range(num_tokens):
        st = int(i * stride)
        if st >= t:
            st = max(0, t - window_size)
        ed = min(t, st + window_size)
        if ed <= st:
            zeros = speed.new_zeros((b,))
            for name in rows:
                rows[name].append(zeros)
            continue

        feats = _activity_distribution_from_window(speed[:, st:ed, :], accel[:, st:ed, :], eps=eps)
        for name in rows:
            rows[name].append(feats[name])

    return {name: torch.stack(vals, dim=1) for name, vals in rows.items()}


def compute_development_dynamic_profile_targets(
    x: torch.Tensor,
    *,
    body_center_idx: int,
    fps: float = 30.0,
    target_feature_names=None,
    window_size: int = 30,
    stride: int = 30,
    num_tokens: int = None,
    profile_bins: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Motif-window temporal activity profile targets.

    Returns [B, num_tokens, profile_bins * F]. Each motif window is split into
    equal contiguous sub-bins. Features are computed from each sub-bin's
    frame x animal movement distribution, preserving timing of bursty motion.
    """
    B, T, _, _, _ = _validate_input_tensor(x, expected_ndim=5, expected_last_dim=2)
    target_names = normalize_behavior_feature_names(target_feature_names or DYNAMIC_PROFILE_FEATURE_NAMES)
    unknown = [name for name in target_names if name not in ACTIVITY_DISTRIBUTION_FEATURE_NAMES]
    if unknown:
        raise ValueError(f"Dynamic profile feature(s) must be activity distribution features, got {unknown}")

    window_size = max(1, int(window_size))
    stride = max(1, int(stride))
    profile_bins = max(1, int(profile_bins))
    if num_tokens is None:
        num_tokens = int((T + stride - 1) // stride)
    num_tokens = max(1, int(num_tokens))

    speed = compute_speed(
        x,
        fps=float(fps),
        body_center_idx=int(body_center_idx),
        per_keypoint=False,
    )
    if T >= 3:
        accel = compute_acceleration(
            x,
            fps=float(fps),
            body_center_idx=int(body_center_idx),
            per_keypoint=False,
        )
    else:
        accel = torch.zeros_like(speed)

    token_rows = []
    for i in range(num_tokens):
        st = int(i * stride)
        if st >= T:
            st = max(0, T - window_size)
        ed = min(T, st + window_size)
        if ed <= st:
            token_rows.append(speed.new_zeros((B, profile_bins, len(target_names))))
            continue
        edges = torch.linspace(st, ed, steps=profile_bins + 1, device=speed.device)
        bin_rows = []
        for bidx in range(profile_bins):
            bst = int(torch.floor(edges[bidx]).item())
            bed = int(torch.floor(edges[bidx + 1]).item())
            if bidx == profile_bins - 1:
                bed = ed
            if bed <= bst:
                bin_rows.append(speed.new_zeros((B, len(target_names))))
            else:
                feats = _activity_distribution_from_window(speed[:, bst:bed, :], accel[:, bst:bed, :], eps=eps)
                bin_rows.append(torch.stack([feats[name] for name in target_names], dim=-1))
        token_rows.append(torch.stack(bin_rows, dim=1))

    profile = torch.stack(token_rows, dim=1)
    return torch.nan_to_num(profile.reshape(B, num_tokens, profile_bins * len(target_names)), nan=0.0, posinf=0.0, neginf=0.0)


def compute_development_behavior_targets(
    x: torch.Tensor,
    *,
    body_center_idx: int,
    fps: float = 30.0,
    keypoint_names=None,
    nose_idx: int = None,
    neck_idx: int = None,
    tail_idx: int = None,
    proximity_tau: float = 1.0,
    synchrony_half_window: int = 15,
    target_feature_names=None,
    activity_gmm_params: dict = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Frame-wise target pack for 30-frame motif supervision.

    Returns [B, T, F] with target_feature_names order.
    All contact/investigation scores are continuous soft scores, not event thresholds.
    """
    B, T, N, K, _ = _validate_input_tensor(x, expected_ndim=5, expected_last_dim=2)
    x = _clean_tensor(x, eps=eps)

    body_center_idx = int(body_center_idx)
    nose_idx = resolve_keypoint_index(keypoint_names, ["Nose"], 0 if nose_idx is None else nose_idx)
    neck_idx = resolve_keypoint_index(keypoint_names, ["Neck"], body_center_idx if neck_idx is None else neck_idx)
    tail_idx = resolve_keypoint_index(keypoint_names, ["Tail", "Tail_base"], body_center_idx if tail_idx is None else tail_idx)
    nose_idx = int(max(0, min(K - 1, nose_idx)))
    neck_idx = int(max(0, min(K - 1, neck_idx)))
    tail_idx = int(max(0, min(K - 1, tail_idx)))

    body = x[:, :, :, body_center_idx, :]
    nose = x[:, :, :, nose_idx, :]
    neck = x[:, :, :, neck_idx, :]
    tail = x[:, :, :, tail_idx, :]

    body_d = torch.cdist(body.reshape(B * T, N, 2), body.reshape(B * T, N, 2), p=2).reshape(B, T, N, N)
    social_spacing = _offdiag_mean(body_d)

    social_distance_delta = torch.zeros_like(social_spacing)
    if T > 1:
        dt_scale = float(fps) if fps and fps > 0 else 1.0
        social_distance_delta[:, 1:] = (social_spacing[:, 1:] - social_spacing[:, :-1]) * dt_scale

    eye = torch.eye(N, device=x.device, dtype=torch.bool).view(1, 1, N, N)
    body_d_no_self = torch.where(eye, torch.tensor(float("nan"), device=x.device, dtype=x.dtype), body_d)
    animal_mean_d = torch.nanmean(body_d_no_self, dim=-1)
    social_isolation = torch.std(animal_mean_d, dim=-1, unbiased=False) / social_spacing.clamp_min(eps)

    scale = torch.linalg.vector_norm(nose - tail, dim=-1).mean(dim=-1).clamp_min(eps)  # [B,T]
    scale_pair = scale.view(B, T, 1, 1)

    head_axis = nose - neck
    fallback_axis = nose - body
    head_norm = torch.linalg.vector_norm(head_axis, dim=-1, keepdim=True)
    fallback_norm = torch.linalg.vector_norm(fallback_axis, dim=-1, keepdim=True)
    head_axis = torch.where(head_norm > eps, head_axis, fallback_axis)
    head_unit = head_axis / torch.linalg.vector_norm(head_axis, dim=-1, keepdim=True).clamp_min(eps)

    nose_i = nose.unsqueeze(3)
    tail_j = tail.unsqueeze(2)
    nose_j = nose.unsqueeze(2)
    head_i = head_unit.unsqueeze(3)

    nt_vec = tail_j - nose_i
    nt_dist = torch.linalg.vector_norm(nt_vec, dim=-1) / scale_pair
    nt_unit = nt_vec / torch.linalg.vector_norm(nt_vec, dim=-1, keepdim=True).clamp_min(eps)
    nt_align = torch.sum(head_i * nt_unit, dim=-1)
    nt_prox = torch.exp(-(nt_dist * nt_dist) / (float(max(proximity_tau, eps)) ** 2))
    anogenital = nt_prox * _soft01_torch(nt_align)

    nn_vec = nose_j - nose_i
    nn_dist = torch.linalg.vector_norm(nn_vec, dim=-1) / scale_pair
    head_head = torch.exp(-(nn_dist * nn_dist) / (float(max(proximity_tau, eps)) ** 2))

    offdiag = (~eye).to(dtype=x.dtype)
    anogenital_investigation = (anogenital * offdiag).sum(dim=(2, 3)) / offdiag.sum(dim=(2, 3)).clamp_min(1.0)
    head_head_contact = (head_head * offdiag).sum(dim=(2, 3)) / offdiag.sum(dim=(2, 3)).clamp_min(1.0)

    speed = compute_speed(
        x,
        fps=float(fps),
        body_center_idx=body_center_idx,
        per_keypoint=False,
    )
    speed_rms = torch.sqrt(torch.mean(speed * speed, dim=-1) + eps)
    speed_mean = torch.mean(speed, dim=-1)
    speed_std = torch.std(speed, dim=-1, unbiased=False)
    activity_imbalance = speed_std / speed_mean.clamp_min(eps)

    if T >= 3:
        accel = compute_acceleration(
            x,
            fps=float(fps),
            body_center_idx=body_center_idx,
            per_keypoint=False,
        )
        accel_rms = torch.sqrt(torch.mean(accel * accel, dim=-1) + eps)
    else:
        accel = torch.zeros_like(speed)
        accel_rms = torch.zeros_like(speed_rms)

    try:
        speed_sync = compute_group_synchronization(
            x,
            body_center_idx=body_center_idx,
            fps=float(fps),
            half_window=int(max(1, synchrony_half_window)),
            use_delta=False,
            min_count=min(5, max(1, T)),
        )
    except Exception:
        speed_sync = torch.zeros_like(speed_rms)
    speed_synchrony = torch.nan_to_num(speed_sync, nan=0.0, posinf=0.0, neginf=0.0)

    feature_map = {
        "social_spacing": social_spacing,
        "social_distance_delta": social_distance_delta,
        "social_isolation": social_isolation,
        "anogenital_investigation": anogenital_investigation,
        "head_head_contact": head_head_contact,
        "speed_rms": speed_rms,
        "accel_rms": accel_rms,
        "activity_imbalance": activity_imbalance,
        "speed_synchrony": speed_synchrony,
    }
    feature_map.update(_frame_activity_distribution_features(speed, accel, eps=eps))
    target_names = normalize_behavior_feature_names(target_feature_names)
    activity_names = {
        "p_inactive",
        "p_active",
        "p_high_activity",
        "p_any_activity",
        "soft_activity_index",
        "state_entropy",
        "active_intensity",
    }
    if any(name in activity_names for name in target_names):
        feature_map.update(
            _activity_gmm_features(
                speed_rms=speed_rms,
                accel_rms=accel_rms,
                activity_imbalance=activity_imbalance,
                params=activity_gmm_params,
                eps=eps,
            )
        )
    missing = [name for name in target_names if name not in feature_map]
    if missing:
        raise ValueError(f"Unknown behavior target feature(s): {missing}")
    target = torch.stack([feature_map[name] for name in target_names], dim=-1)
    return torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)


def compute_development_behavior_window_targets(
    x: torch.Tensor,
    *,
    body_center_idx: int,
    fps: float = 30.0,
    keypoint_names=None,
    nose_idx: int = None,
    neck_idx: int = None,
    tail_idx: int = None,
    proximity_tau: float = 1.0,
    synchrony_half_window: int = 15,
    target_feature_names=None,
    activity_gmm_params: dict = None,
    window_size: int = 30,
    stride: int = 30,
    num_tokens: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Window-level behavior targets for motif supervision.

    Most targets are frame-wise scores averaged within each motif window.
    Activity-distribution targets are computed directly from the full
    window x animal speed/acceleration distribution to avoid diluting bursty
    movement through an animal mean followed by a temporal mean.
    """
    B, T, N, K, _ = _validate_input_tensor(x, expected_ndim=5, expected_last_dim=2)
    target_names = normalize_behavior_feature_names(target_feature_names)
    window_size = max(1, int(window_size))
    stride = max(1, int(stride))
    if num_tokens is None:
        num_tokens = int((T + stride - 1) // stride)
    num_tokens = max(1, int(num_tokens))

    frame_target = compute_development_behavior_targets(
        x,
        body_center_idx=body_center_idx,
        fps=fps,
        keypoint_names=keypoint_names,
        nose_idx=nose_idx,
        neck_idx=neck_idx,
        tail_idx=tail_idx,
        proximity_tau=proximity_tau,
        synchrony_half_window=synchrony_half_window,
        target_feature_names=target_names,
        activity_gmm_params=activity_gmm_params,
        eps=eps,
    )

    chunks = []
    for i in range(num_tokens):
        st = int(i * stride)
        if st >= T:
            st = max(0, T - window_size)
        ed = min(T, st + window_size)
        if ed <= st:
            chunks.append(frame_target.new_zeros((B, len(target_names))))
        else:
            chunks.append(frame_target[:, st:ed, :].mean(dim=1))
    window_target = torch.stack(chunks, dim=1)

    if any(name in ACTIVITY_DISTRIBUTION_FEATURE_NAMES for name in target_names):
        speed = compute_speed(
            x,
            fps=float(fps),
            body_center_idx=int(body_center_idx),
            per_keypoint=False,
        )
        if T >= 3:
            accel = compute_acceleration(
                x,
                fps=float(fps),
                body_center_idx=int(body_center_idx),
                per_keypoint=False,
            )
        else:
            accel = torch.zeros_like(speed)
        dist_features = _window_activity_distribution_features(
            speed,
            accel,
            window_size=window_size,
            stride=stride,
            num_tokens=num_tokens,
            eps=eps,
        )
        for fi, name in enumerate(target_names):
            if name in dist_features:
                window_target[:, :, fi] = dist_features[name]

    return torch.nan_to_num(window_target, nan=0.0, posinf=0.0, neginf=0.0)


def compute_group_relative_orientation(
    x: torch.Tensor,
    *,
    mask: torch.Tensor = None,
    nose_idx: int,
    neck_idx: int,
    body_center_idx: int,
    heading_source: str = "head",
    smooth_window: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:

    RO_matrix = compute_relative_orientation(
        x,
        mask=mask,
        nose_idx=nose_idx,
        neck_idx=neck_idx,
        body_center_idx=body_center_idx,
        heading_source=heading_source,
        smooth_window=smooth_window,
        eps=eps,
    )

    RO_sym = 0.5 * (RO_matrix + RO_matrix.transpose(-2, -1))
    
    return _compute_pairwise_mean(RO_sym, mask, body_center_idx, eps)


def compute_group_heading_alignment(
    x: torch.Tensor,
    *,
    mask: torch.Tensor = None,
    nose_idx: int,
    neck_idx: int,
    body_center_idx: int,
    heading_source: str = "head",
    smooth_window: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    align = compute_heading_alignment(
        x,
        mask=mask,
        nose_idx=nose_idx,
        neck_idx=neck_idx,
        body_center_idx=body_center_idx,
        heading_source=heading_source,
        smooth_window=smooth_window,
        eps=eps,
    )
    return _compute_pairwise_mean(align, mask, body_center_idx, eps)


def compute_group_reciprocal_attention_strength(
    x: torch.Tensor,
    *,
    mask: torch.Tensor = None,
    nose_idx: int,
    neck_idx: int,
    body_center_idx: int,
    heading_source: str = "head",
    smooth_window: int = 1,
    cone_angle_deg: float = 60.0,
    attention_temp: float = 0.15,
    distance_tau: float = 0.6,
    eps: float = 1e-6,
) -> torch.Tensor:
    attn = compute_directed_attention_scores(
        x,
        mask=mask,
        nose_idx=nose_idx,
        neck_idx=neck_idx,
        body_center_idx=body_center_idx,
        heading_source=heading_source,
        smooth_window=smooth_window,
        cone_angle_deg=cone_angle_deg,
        attention_temp=attention_temp,
        distance_tau=distance_tau,
        eps=eps,
    )
    reciprocal = attn * attn.transpose(-2, -1)
    return _compute_pairwise_mean(reciprocal, mask, body_center_idx, eps)


def compute_group_shared_attention_strength(
    x: torch.Tensor,
    *,
    mask: torch.Tensor = None,
    nose_idx: int,
    neck_idx: int,
    body_center_idx: int,
    heading_source: str = "head",
    smooth_window: int = 1,
    cone_angle_deg: float = 60.0,
    attention_temp: float = 0.15,
    distance_tau: float = 0.6,
    eps: float = 1e-6,
) -> torch.Tensor:
    attn = compute_directed_attention_scores(
        x,
        mask=mask,
        nose_idx=nose_idx,
        neck_idx=neck_idx,
        body_center_idx=body_center_idx,
        heading_source=heading_source,
        smooth_window=smooth_window,
        cone_angle_deg=cone_angle_deg,
        attention_temp=attention_temp,
        distance_tau=distance_tau,
        eps=eps,
    )
    B, T, N, _ = attn.shape
    shared_terms = []
    for target in range(N):
        sources = [idx for idx in range(N) if idx != target]
        for src_i in range(len(sources)):
            for src_j in range(src_i + 1, len(sources)):
                shared_terms.append(attn[..., sources[src_i], target] * attn[..., sources[src_j], target])
    if not shared_terms:
        return torch.zeros((B, T), device=attn.device, dtype=attn.dtype)
    return torch.stack(shared_terms, dim=-1).mean(dim=-1)
