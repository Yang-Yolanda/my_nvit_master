#!/home/yangz/.conda/envs/4D-humans/bin/python
import numpy as np
import torch

# SMPL Joint Names (24 joints)
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
]

# Parent-Child Relationships (Directed Edges)
# Format: (Parent, Child)
SMPL_PARENTS = {
    'pelvis': None,
    'left_hip': 'pelvis',
    'right_hip': 'pelvis',
    'spine1': 'pelvis',
    'left_knee': 'left_hip',
    'right_knee': 'right_hip',
    'spine2': 'spine1',
    'left_ankle': 'left_knee',
    'right_ankle': 'right_knee',
    'spine3': 'spine2',
    'left_foot': 'left_ankle',
    'right_foot': 'right_ankle',
    'neck': 'spine3',
    'left_collar': 'spine3',
    'right_collar': 'spine3',
    'head': 'neck',
    'left_shoulder': 'left_collar',
    'right_shoulder': 'right_collar',
    'left_elbow': 'left_shoulder',
    'right_elbow': 'right_shoulder',
    'left_wrist': 'left_elbow',
    'right_wrist': 'right_elbow',
    'left_hand': 'left_wrist',
    'right_hand': 'right_wrist'
}

def get_smpl_adjacency_matrix(directed=True, add_self_loops=True):
    """
    Generates the Adjacency Matrix for the SMPL skeleton.
    
    Args:
        directed (bool): If True, edges only go Parent -> Child (State Transition).
                        If False, edges are bidirectional (Standard GCN).
        add_self_loops (bool): If True, adds diagonal I to A.
    
    Returns:
        torch.Tensor: [24, 24] Adjacency Matrix
    """
    num_joints = len(SMPL_JOINT_NAMES)
    name_to_idx = {name: i for i, name in enumerate(SMPL_JOINT_NAMES)}
    
    adj = torch.zeros((num_joints, num_joints), dtype=torch.float32)
    
    for child, parent in SMPL_PARENTS.items():
        if parent is None:
            continue
        
        p_idx = name_to_idx[parent]
        c_idx = name_to_idx[child]
        
        # Edge: Parent -> Child
        adj[p_idx, c_idx] = 1.0
        
        if not directed:
            # Undirected: Child -> Parent as well
            adj[c_idx, p_idx] = 1.0
            
    if add_self_loops:
        adj = adj + torch.eye(num_joints)
        
    return adj

def get_k_hop_adjacency(k=2, directed=True, add_self_loops=True):
    """
    Generates an Adjacency Matrix including neighbors up to K hops away.
    Useful for "soft" kinematic constraints (e.g. Shoulder can see Wrist).
    
    Args:
        k (int): Number of hops. k=1 is standard adjacency.
        directed (bool): Single direction flow.
    """
    base_adj = get_smpl_adjacency_matrix(directed=directed, add_self_loops=add_self_loops)
    
    # K-hop via matrix power (or Boolean multiplication)
    # A^k tells us reachability in k steps
    
    # We want union of 1..k hops: (A + A^2 + ... + A^k) > 0
    k_hop_adj = base_adj.clone()
    curr_power = base_adj.clone()
    
    for _ in range(k - 1):
        curr_power = torch.matmul(curr_power, base_adj)
        k_hop_adj = k_hop_adj + curr_power
        
    # Binarize
    return (k_hop_adj > 0).float()

def get_geodesic_distance_matrix(directed=False):
    """
    Computes the Geodesic Distance Matrix (Shortest Path Steps) between joints.
    
    Args:
        directed (bool): If True, distances only follow Parent -> Child paths.
                        If False, any path in the tree is allowed.
    
    Returns:
        torch.Tensor: [24, 24] Distance Matrix
    """
    num_joints = len(SMPL_JOINT_NAMES)
    adj = get_smpl_adjacency_matrix(directed=directed, add_self_loops=False)
    
    # Initialize distances to infinity
    dist = torch.full((num_joints, num_joints), float('inf'))
    dist.fill_diagonal_(0)
    dist[adj > 0] = 1.0
    
    # Floyd-Warshall Algorithm
    for k in range(num_joints):
        for i in range(num_joints):
            for j in range(num_joints):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    
    return dist

def get_kinematic_chain_order():
    """
    Returns a list of joint indices sorted by depth (Topological Sort).
    Essential for sequential state updates.
    """
    # Simple BFS/Level-order traversal from Pelvis
    name_to_idx = {name: i for i, name in enumerate(SMPL_JOINT_NAMES)}
    
    # Calculate depth for sorting
    depths = {'pelvis': 0}
    queue = ['pelvis']
    
    level_ordered_indices = []
    
    while queue:
        curr = queue.pop(0)
        level_ordered_indices.append(name_to_idx[curr])
        
        # Find children
        children = [c for c, p in SMPL_PARENTS.items() if p == curr]
        for child in children:
            depths[child] = depths[curr] + 1
            queue.append(child)
            
    return level_ordered_indices

if __name__ == "__main__":
    A = get_smpl_adjacency_matrix(directed=True)
    print("Directed Adjacency Matrix shape:", A.shape)
    print("Non-zero entries (Edges + Self-loops):", torch.nonzero(A).shape[0])
    
    order = get_kinematic_chain_order()
    print("Topological Processing Order:", order)
