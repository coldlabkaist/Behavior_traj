"""Published analysis core."""
import os

# Preserve the original GMM scripts' worker setting on Windows.
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
