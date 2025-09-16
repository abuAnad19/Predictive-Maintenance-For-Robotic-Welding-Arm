from collections import deque

FAULT_GRAPH = {
    "anomaly": ["mechanical", "process"],
    "mechanical": ["bearing_wear", "misalignment"],
    "process": ["gas_issue", "tip_wear"],
    "bearing_wear": [],
    "misalignment": [],
    "gas_issue": [],
    "tip_wear": [],
}

def bfs_fault_path(start="anomaly", target=None):
    visited = set([start])
    q = deque([(start, [start])])
    while q:
        node, path = q.popleft()
        if target is None and not FAULT_GRAPH.get(node):
            return path
        if target and node == target:
            return path
        for nxt in FAULT_GRAPH.get(node, []):
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, path + [nxt]))
    return []
