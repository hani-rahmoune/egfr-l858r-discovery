"""
Tests for Phase 13: GNN featurizer and forward pass.

Coverage:
  TestFeaturizer   — atom/bond feature shapes, known SMILES, invalid inputs
  TestBuildModel   — model construction, param count, forward pass shape
  TestTrainStep    — one gradient step, loss decreases, eval mode

PyTorch and torch_geometric are required; tests are skipped if absent.
No checkpoint files needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# All tests in this file need torch_geometric; skip the whole module if absent.
torch = pytest.importorskip("torch", reason="PyTorch not installed")
pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from src.models.gnn_models import (
    N_ATOM_FEATS,
    N_BOND_FEATS,
    atom_features,
    bond_features,
    build_gin_predictor,
    featurize,
    featurize_batch,
)

# ── Test molecules ─────────────────────────────────────────────────────────────

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
BENZENE = "c1ccccc1"
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
METHANE = "C"  # single atom, no bonds — edge case
INVALID = "not_smiles_$$"


# ── TestFeaturizer ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestFeaturizer:
    def test_atom_features_length(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(ASPIRIN)
        feats = atom_features(mol.GetAtomWithIdx(0))
        assert len(feats) == N_ATOM_FEATS, f"expected {N_ATOM_FEATS}, got {len(feats)}"

    def test_atom_features_binary(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(BENZENE)
        for atom in mol.GetAtoms():
            feats = atom_features(atom)
            assert all(v in (0, 1) for v in feats), "atom features must be 0 or 1"

    def test_bond_features_length(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(BENZENE)
        for bond in mol.GetBonds():
            feats = bond_features(bond)
            assert (
                len(feats) == N_BOND_FEATS
            ), f"expected {N_BOND_FEATS}, got {len(feats)}"

    def test_bond_features_binary(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(ASPIRIN)
        for bond in mol.GetBonds():
            for v in bond_features(bond):
                assert v in (0, 1)

    def test_featurize_aspirin(self):
        data = featurize(ASPIRIN, y=7.0)
        assert data is not None
        assert data.x.shape[1] == N_ATOM_FEATS
        assert data.edge_attr.shape[1] == N_BOND_FEATS
        assert data.edge_index.shape[0] == 2
        # Edges are undirected: 2× bonds
        from rdkit import Chem

        mol = Chem.MolFromSmiles(ASPIRIN)
        n_bonds = mol.GetNumBonds()
        assert data.edge_index.shape[1] == n_bonds * 2
        assert data.y is not None
        assert float(data.y) == pytest.approx(7.0)

    def test_featurize_no_label(self):
        data = featurize(BENZENE)
        assert data is not None
        assert data.y is None

    def test_featurize_invalid_smiles(self):
        data = featurize(INVALID)
        assert data is None, "invalid SMILES should return None"

    def test_featurize_methane_self_loop(self):
        # Single atom with no bonds — should get self-loop padding, not crash
        data = featurize(METHANE)
        assert data is not None
        assert data.x.shape == (1, N_ATOM_FEATS)
        # Self-loop: 1 edge
        assert data.edge_index.shape[1] >= 1

    def test_featurize_batch(self):
        smiles = [ASPIRIN, GEFITINIB, INVALID, BENZENE]
        ys = [7.0, 8.5, 0.0, 6.0]
        data_list, valid_idx = featurize_batch(smiles, ys)
        # INVALID should be skipped
        assert len(data_list) == 3
        assert INVALID not in [smiles[i] for i in valid_idx]
        assert 0 in valid_idx and 1 in valid_idx and 3 in valid_idx

    def test_featurize_batch_no_labels(self):
        smiles = [ASPIRIN, BENZENE]
        data_list, valid_idx = featurize_batch(smiles)
        assert len(data_list) == 2
        assert all(d.y is None for d in data_list)

    def test_gefitinib_atom_count(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(GEFITINIB)
        data = featurize(GEFITINIB)
        assert data is not None
        assert data.x.shape[0] == mol.GetNumAtoms()


# ── TestBuildModel ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildModel:
    def _build_small(self):
        return build_gin_predictor(
            in_channels=N_ATOM_FEATS,
            edge_dim=N_BOND_FEATS,
            hidden_channels=32,
            num_layers=2,
            dropout=0.0,
        )

    def test_model_is_module(self):
        import torch.nn as nn

        model = self._build_small()
        assert isinstance(model, nn.Module)

    def test_param_count_positive(self):
        model = self._build_small()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "model must have learnable parameters"

    def test_forward_shape_single(self):
        from torch_geometric.data import Batch

        model = self._build_small()
        data = featurize(ASPIRIN, y=7.0)
        batch = Batch.from_data_list([data])
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert out.shape == (1,), f"expected shape (1,), got {out.shape}"

    def test_forward_shape_batch(self):
        from torch_geometric.data import Batch

        model = self._build_small()
        data_list = [
            featurize(s, y=float(i))
            for i, s in enumerate([ASPIRIN, BENZENE, GEFITINIB])
        ]
        batch = Batch.from_data_list(data_list)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert out.shape == (3,), f"expected (3,), got {out.shape}"

    def test_output_is_finite(self):
        from torch_geometric.data import Batch

        model = self._build_small()
        batch = Batch.from_data_list([featurize(GEFITINIB, y=8.0)])
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert torch.isfinite(out).all(), "model output contains NaN/Inf"

    def test_deterministic_in_eval_mode(self):
        """Two identical forward passes in eval mode must produce the same output."""
        from torch_geometric.data import Batch

        model = build_gin_predictor(
            in_channels=N_ATOM_FEATS,
            edge_dim=N_BOND_FEATS,
            hidden_channels=16,
            num_layers=1,
            dropout=0.1,
        )
        batch = Batch.from_data_list([featurize(ASPIRIN, y=7.0)])
        model.eval()
        with torch.no_grad():
            out1 = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            out2 = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert torch.allclose(out1, out2), "eval-mode output is not deterministic"


# ── TestTrainStep ──────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestTrainStep:
    def _setup(self):
        from torch_geometric.loader import DataLoader

        smiles_list = [ASPIRIN, BENZENE, GEFITINIB, METHANE, ASPIRIN]
        ys = [7.0, 5.5, 8.5, 4.0, 7.2]
        data_list, _ = featurize_batch(smiles_list, ys)
        loader = DataLoader(data_list, batch_size=4, shuffle=False)
        model = build_gin_predictor(
            in_channels=N_ATOM_FEATS,
            edge_dim=N_BOND_FEATS,
            hidden_channels=32,
            num_layers=2,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return model, loader, optimizer

    def test_loss_is_finite(self):
        import torch.nn.functional as F

        model, loader, optimizer = self._setup()
        model.train()
        for batch in loader:
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.mse_loss(out, batch.y.squeeze(-1))
            assert torch.isfinite(loss), f"loss is not finite: {loss}"
            break

    def test_gradients_flow(self):
        import torch.nn.functional as F

        model, loader, optimizer = self._setup()
        model.train()
        for batch in loader:
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.mse_loss(out, batch.y.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            break
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(
            g is not None and g.abs().max() > 0 for g in grads
        ), "no gradients flowed"

    def test_loss_decreases_over_ten_steps(self):
        """Loss should decrease over 10 gradient steps on a tiny batch."""
        import torch.nn.functional as F
        from torch_geometric.loader import DataLoader

        # Tiny fixed set
        smiles_list = [ASPIRIN, GEFITINIB, BENZENE, METHANE, ASPIRIN, GEFITINIB]
        ys = [7.0, 8.5, 5.5, 4.0, 7.0, 8.5]
        data_list, _ = featurize_batch(smiles_list, ys)
        loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)

        model = build_gin_predictor(
            in_channels=N_ATOM_FEATS,
            edge_dim=N_BOND_FEATS,
            hidden_channels=32,
            num_layers=2,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        model.train()

        losses = []
        for _ in range(10):
            for batch in loader:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = F.mse_loss(out, batch.y.squeeze(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        # Loss over last step should be lower than over first step
        assert (
            losses[-1] < losses[0]
        ), f"loss did not decrease: first={losses[0]:.4f} last={losses[-1]:.4f}"
