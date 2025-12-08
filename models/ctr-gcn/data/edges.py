# src/ctr-gcn/data/edges.py

# NTU RGB+D 25 joints edges (u, v)
# ※論文と NTU の関節仕様に基づく
EDGES = [
    (1, 0), (2, 1), (3, 2),           # Spine
    (4, 3), (5, 4), (6, 5),           # Left arm
    (7, 3), (8, 7), (9, 8),           # Right arm
    (10, 0), (11, 10), (12, 11),      # Left leg
    (13, 0), (14, 13), (15, 14),      # Right leg
    (16, 5), (17, 6),                 # Left hand
    (18, 9), (19, 8),                 # Right hand
    (20, 12), (21, 12), (22, 15), (23, 15)
]
