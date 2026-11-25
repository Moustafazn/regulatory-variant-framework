"""Operational effect classes and indivisible genomic locus groups."""
import numpy as np
CLASSES = ['maintained', 'weakened', 'strengthened', 'reversed']


def effect_classes(source, target, margin):
    # Assign operational point classes; non-significance is not equivalence.
    source, target = np.broadcast_arrays(np.asarray(source), np.asarray(target))
    changed = np.abs(target - source) > margin
    reverse = (source * target < 0) & (np.minimum(np.abs(source), np.abs(target)) > margin)
    label = np.where(np.abs(target) > np.abs(source), 2, 1)
    return np.where(changed, np.where(reverse, 3, label), 0).astype(int)



class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size)

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return int(item)

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def locus_groups(frame, gap_bp=250_000):
    # Join nearby variants and shared identifiers into indivisible leakage groups.
    frame = frame.reset_index(drop=True)
    links = UnionFind(len(frame))
    for _, part in frame.groupby(['genome_build', 'chrom'], sort=True):
        order = part.sort_values('position').index.to_numpy()
        for a, b in zip(order[:-1], order[1:]):
            if frame.at[b, 'position'] - frame.at[a, 'position'] <= gap_bp:
                links.union(a, b)
    for column in ['variant_id', 'rsid', 'lead_group']:
        if column not in frame.columns:
            continue
        for _, part in frame.dropna(subset=[column]).groupby(['genome_build', column]):
            indices = part.index.to_numpy()
            for item in indices[1:]:
                links.union(indices[0], item)
    return np.array([f'locus_{links.find(i):07d}' for i in range(len(frame))])
