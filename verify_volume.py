# Quick check - what are the actual volume dimensions?

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from pathlib import Path

npz_files = list(Path("/home/pahm409/isles2022_preprocessed").glob("*.npz"))[:5]

for f in npz_files:
    data = np.load(f)
    print(f"{f.stem}: shape={data['image'].shape}")
