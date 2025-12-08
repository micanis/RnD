# src/ctr-gcn/data/ntu_reader.py
from pathlib import Path
import numpy as np
from config import NTU_V, MAX_M

def read_skeleton(fp: Path):
    frames = []
    with fp.open("r") as f:
        T = int(f.readline())
        for _ in range(T):
            n_bodies = int(f.readline())
            persons = []
            for _ in range(n_bodies):
                f.readline()  # skip bodyID meta
                v = int(f.readline())
                joints = []
                for _ in range(v):
                    x, y, z = list(map(float, f.readline().split()[:3]))
                    joints.append([x, y, z])
                persons.append(np.array(joints, dtype=np.float32))

            if len(persons) == 0:
                persons = [np.zeros((NTU_V, 3), np.float32)]

            persons = persons[:MAX_M] + [np.zeros((NTU_V, 3), np.float32)] * (MAX_M - len(persons))
            frames.append(np.stack(persons, axis=0))  # (M,V,3)
    return frames
