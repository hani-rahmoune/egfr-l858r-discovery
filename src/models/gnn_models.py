"""
Molecular GNN for pIC50 regression.

Architecture: atom/bond featurization → linear embedding → GINEConv × L layers
(with BatchNorm, ReLU, dropout) → global mean pooling → MLP head.

GINEConv (Hu et al. 2020) propagates both node and edge features:
    x_i' = MLP((1+ε)·x_i + Σ_{j∈N(i)} ReLU(x_j + e_{ij}))
where e_{ij} is projected to the same width as x.

Public API
----------
    featurize(smiles)  ->  torch_geometric.data.Data | None
    GINPredictor(in_channels, edge_dim, **kwargs)

Atom feature vector (N_ATOM_FEATS = 41):
    atomic_num one-hot  [H,B,C,N,O,F,Si,P,S,Cl,Br,I,other]  13
    degree one-hot      [0,1,2,3,4,5,6,other]                  8
    hybridization       [SP,SP2,SP3,SP3D,SP3D2,other]          6
    formal_charge       [-2,-1,0,1,2,other]                    6
    num_hs              [0,1,2,3,4,other]                      6
    is_aromatic                                                 1
    is_in_ring                                                  1
                                                Total:         41

Bond feature vector (N_BOND_FEATS = 6):
    bond_type one-hot   [SINGLE,DOUBLE,TRIPLE,AROMATIC]         4
    is_conjugated                                               1
    is_in_ring                                                  1
                                                Total:          6
"""

from __future__ import annotations

N_ATOM_FEATS = 41
N_BOND_FEATS = 6

# ── One-hot helper ─────────────────────────────────────────────────────────────

_ATOMIC_NUMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H B C N O F Si P S Cl Br I
_DEGREES = [0, 1, 2, 3, 4, 5, 6]
_HYBRIDIZATIONS: list  # filled below after rdkit import
_FORMAL_CHARGES = [-2, -1, 0, 1, 2]
_NUM_HS = [0, 1, 2, 3, 4]


def _one_hot(val, values: list) -> list[int]:
    """One-hot encoding with an extra 'other' bucket at the end."""
    vec = [0] * (len(values) + 1)
    try:
        vec[values.index(val)] = 1
    except ValueError:
        vec[-1] = 1  # 'other'
    return vec


# ── Atom featurizer ────────────────────────────────────────────────────────────


def atom_features(atom) -> list[int]:
    """Build the 41-dim atom feature vector for an RDKit atom."""
    from rdkit.Chem import rdchem

    hybrid_map = {
        rdchem.HybridizationType.SP: 0,
        rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2,
        rdchem.HybridizationType.SP3D: 3,
        rdchem.HybridizationType.SP3D2: 4,
    }
    hybrid_idx = hybrid_map.get(atom.GetHybridization(), 5)
    hybrid_oh = [0] * 6
    hybrid_oh[hybrid_idx] = 1

    return (
        _one_hot(atom.GetAtomicNum(), _ATOMIC_NUMS)  # 13
        + _one_hot(atom.GetDegree(), _DEGREES)  # 8
        + hybrid_oh  # 6
        + _one_hot(atom.GetFormalCharge(), _FORMAL_CHARGES)  # 6
        + _one_hot(atom.GetTotalNumHs(), _NUM_HS)  # 6
        + [int(atom.GetIsAromatic())]  # 1
        + [int(atom.IsInRing())]  # 1
    )


# ── Bond featurizer ────────────────────────────────────────────────────────────


def bond_features(bond) -> list[int]:
    """Build the 6-dim bond feature vector for an RDKit bond."""
    from rdkit.Chem import rdchem

    bt = bond.GetBondType()
    bond_type_oh = [
        int(bt == rdchem.BondType.SINGLE),
        int(bt == rdchem.BondType.DOUBLE),
        int(bt == rdchem.BondType.TRIPLE),
        int(bt == rdchem.BondType.AROMATIC),
    ]
    return bond_type_oh + [int(bond.GetIsConjugated()), int(bond.IsInRing())]


# ── SMILES → PyG Data ─────────────────────────────────────────────────────────


def featurize(smiles: str, y: float | None = None):
    """
    Convert a SMILES string to a torch_geometric.data.Data object.

    Returns None for invalid SMILES or molecules with no bonds
    (single atoms cannot form a graph with edges).
    """
    import torch
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if mol.GetNumAtoms() == 0:
        return None

    # Node features
    xs = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(xs, dtype=torch.float)  # [n_atoms, N_ATOM_FEATS]

    # Edge index and edge features (undirected → add both directions)
    row, col, edge_attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        row += [i, j]
        col += [j, i]
        edge_attrs += [bf, bf]

    if not row:
        # No bonds — pad with self-loops so the graph has edges
        n = mol.GetNumAtoms()
        row = list(range(n))
        col = list(range(n))
        edge_attrs = [[1, 0, 0, 0, 0, 0]] * n  # SINGLE self-loop placeholder

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)  # [n_bonds*2, N_BOND_FEATS]

    label = torch.tensor([y], dtype=torch.float) if y is not None else None
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label)


def featurize_batch(
    smiles_list: list[str],
    y_list: list[float] | None = None,
) -> tuple[list, list[int]]:
    """
    Featurize a list of SMILES strings.
    Returns (data_list, valid_indices) — invalid SMILES are silently skipped.
    """
    data_list, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        y = y_list[i] if y_list is not None else None
        d = featurize(smi, y=y)
        if d is not None:
            data_list.append(d)
            valid_idx.append(i)
    return data_list, valid_idx


# ── GINEConv predictor ────────────────────────────────────────────────────────


def _gin_mlp(width: int):
    """Two-layer MLP used inside each GINEConv layer."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(width, width * 2),
        nn.ReLU(),
        nn.Linear(width * 2, width),
    )


class GINPredictor:
    """
    Placeholder sentinel — import the real class as:
        from src.models.gnn_models import build_gin_predictor

    This exists so other modules can do ``GINPredictor`` type checks without
    importing torch at module load time.
    """


def build_gin_predictor(
    in_channels: int = N_ATOM_FEATS,
    edge_dim: int = N_BOND_FEATS,
    hidden_channels: int = 128,
    num_layers: int = 4,
    dropout: float = 0.2,
):
    """
    Build the GINPredictor model.  Requires torch and torch_geometric.

    Parameters
    ----------
    in_channels     : atom feature dimension (default N_ATOM_FEATS = 41)
    edge_dim        : bond feature dimension (default N_BOND_FEATS = 6)
    hidden_channels : width used in all GINEConv layers and the MLP head
    num_layers      : number of GINEConv message-passing steps
    dropout         : dropout rate applied after each BN+ReLU block
    """
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import BatchNorm, GINEConv, global_mean_pool

    class _GINPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.atom_emb = nn.Linear(in_channels, hidden_channels)
            self.edge_emb = nn.Linear(edge_dim, hidden_channels)
            self.convs = nn.ModuleList(
                [
                    GINEConv(_gin_mlp(hidden_channels), train_eps=True)
                    for _ in range(num_layers)
                ]
            )
            self.bns = nn.ModuleList(
                [BatchNorm(hidden_channels) for _ in range(num_layers)]
            )
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 1),
            )

        def forward(self, x, edge_index, edge_attr, batch):
            x = F.relu(self.atom_emb(x))
            e = F.relu(self.edge_emb(edge_attr))
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index, e)
                x = bn(x)
                x = F.relu(x)
                x = self.drop(x)
            x = global_mean_pool(x, batch)
            return self.head(x).squeeze(-1)

    return _GINPredictor()
