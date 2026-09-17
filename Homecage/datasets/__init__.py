"""Pose/model-input loaders; original data is resolved from the shared root dataset."""
from datasets.pose_io import PoseReader
from datasets.dataset import MousePoseDataset

__all__ = ['PoseReader', 'MousePoseDataset']
