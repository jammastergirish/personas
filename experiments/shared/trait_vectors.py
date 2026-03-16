"""
Trait vector extraction and projection utilities for persona experiments.

Provides functions to compute per-persona and global trait vectors from hidden
states, measure cross-persona consistency via cosine similarity, compute
residuals, project onto trait bases, and perform SVD analysis.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Add parent paths so we can import from main.py and steer.py
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ============================================================
# Per-persona trait vectors
# ============================================================


def compute_trait_vectors_per_persona(
    by_layer: Dict[int, List[torch.Tensor]],
    persona_names: List[str],
    trait_labels: List[str],
    trait_names: List[str],
    personas: List[str],
    traits: List[str],
) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Compute trait vectors per persona per layer.

    by_layer: {layer: [tensor, ...]} -- hidden states from forward passes
    persona_names: persona name for each example (parallel to by_layer values)
    trait_labels: "high" or "low" for each example
    trait_names: trait name for each example
    personas: ordered list of persona names
    traits: ordered list of trait names

    Returns: {layer: {persona: {trait: vector}}}
    where vector = centroid(high_trait_in_persona) - centroid(low_trait_in_persona)
    """
    result: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}

    for layer, vec_list in by_layer.items():
        per_persona: Dict[str, Dict[str, torch.Tensor]] = {}

        for persona in personas:
            per_trait: Dict[str, torch.Tensor] = {}

            for trait in traits:
                high_vecs: List[torch.Tensor] = []
                low_vecs: List[torch.Tensor] = []

                for i, vec in enumerate(vec_list):
                    if persona_names[i] != persona or trait_names[i] != trait:
                        continue
                    if trait_labels[i] == "high":
                        high_vecs.append(vec)
                    elif trait_labels[i] == "low":
                        low_vecs.append(vec)

                if high_vecs and low_vecs:
                    high_centroid = torch.stack(high_vecs, dim=0).mean(dim=0)
                    low_centroid = torch.stack(low_vecs, dim=0).mean(dim=0)
                    per_trait[trait] = high_centroid - low_centroid
                else:
                    # If we lack examples for one side, return a zero vector
                    d = vec_list[0].shape[-1]
                    per_trait[trait] = torch.zeros(d)

            per_persona[persona] = per_trait

        result[layer] = per_persona

    return result


# ============================================================
# Global trait vectors
# ============================================================


def compute_global_trait_vectors(
    trait_vectors_per_persona: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    personas: List[str],
    traits: List[str],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Global trait vector = mean over personas of per-persona trait vectors.

    Returns: {layer: {trait: vector}}
    """
    result: Dict[int, Dict[str, torch.Tensor]] = {}

    for layer, per_persona in trait_vectors_per_persona.items():
        global_traits: Dict[str, torch.Tensor] = {}

        for trait in traits:
            vecs = [
                per_persona[persona][trait]
                for persona in personas
                if trait in per_persona.get(persona, {})
            ]
            if vecs:
                global_traits[trait] = torch.stack(vecs, dim=0).mean(dim=0)
            else:
                raise ValueError(
                    f"No persona vectors found for trait {trait!r} at layer {layer}"
                )

        result[layer] = global_traits

    return result


# ============================================================
# Cross-persona cosine similarity
# ============================================================


def cross_persona_cosine_similarity(
    trait_vectors_per_persona: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    layer: int,
    personas: List[str],
    traits: List[str],
) -> Dict[str, float]:
    """
    For each trait, compute mean pairwise cosine similarity of that trait's
    vector across all persona pairs.

    Returns: {trait: mean_cosine}
    """
    per_persona = trait_vectors_per_persona[layer]
    result: Dict[str, float] = {}

    for trait in traits:
        cosines: List[float] = []
        for p_a, p_b in combinations(personas, 2):
            vec_a = per_persona[p_a][trait]
            vec_b = per_persona[p_b][trait]
            cos = F.cosine_similarity(
                vec_a.unsqueeze(0), vec_b.unsqueeze(0)
            ).item()
            cosines.append(cos)

        result[trait] = float(np.mean(cosines)) if cosines else 0.0

    return result


# ============================================================
# Trait residuals
# ============================================================


def compute_trait_residuals(
    trait_vectors_per_persona: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    global_trait_vectors: Dict[int, Dict[str, torch.Tensor]],
    layer: int,
    personas: List[str],
    traits: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Residual = persona-specific trait vec - global trait vec.

    Returns: {persona: {trait: residual_norm_ratio}}
    where ratio = ||residual|| / ||global||.
    """
    per_persona = trait_vectors_per_persona[layer]
    global_vecs = global_trait_vectors[layer]
    result: Dict[str, Dict[str, float]] = {}

    for persona in personas:
        ratios: Dict[str, float] = {}
        for trait in traits:
            persona_vec = per_persona[persona][trait]
            global_vec = global_vecs[trait]
            residual = persona_vec - global_vec

            global_norm = global_vec.norm().item()
            if global_norm > 0:
                ratios[trait] = residual.norm().item() / global_norm
            else:
                ratios[trait] = 0.0

        result[persona] = ratios

    return result


# ============================================================
# Projection onto trait basis
# ============================================================


def project_onto_trait_basis(
    hidden_states: torch.Tensor,
    trait_vectors: Dict[str, torch.Tensor],
    traits: List[str],
) -> torch.Tensor:
    """
    Project hidden states onto trait basis vectors.

    hidden_states: [n, d]
    trait_vectors: {trait: [d]}
    traits: ordered list of trait names (determines column order)

    Returns: [n, n_traits] -- coordinates in trait space
    """
    # Build the basis matrix [n_traits, d], normalized
    basis_vecs = []
    for trait in traits:
        vec = trait_vectors[trait].float()
        norm = vec.norm()
        if norm > 0:
            vec = vec / norm
        basis_vecs.append(vec)

    basis = torch.stack(basis_vecs, dim=0)  # [n_traits, d]
    h = hidden_states.float()  # [n, d]

    # Project: each row of result is the dot product with each basis vector
    projections = h @ basis.T  # [n, n_traits]

    return projections


# ============================================================
# SVD analysis
# ============================================================


def svd_analysis(
    matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform SVD and return U, singular values, and cumulative variance ratio.

    matrix: [n, d]

    Returns:
        U: [n, k] left singular vectors (k = min(n, d))
        singular_values: [k] singular values in descending order
        cumulative_variance_ratio: [k] cumulative fraction of total variance
            explained by the first k components
    """
    # Center the matrix
    matrix = matrix.float()
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean

    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    # Variance explained by each component is proportional to S^2
    variance = S ** 2
    total_variance = variance.sum()
    if total_variance > 0:
        cumulative_variance_ratio = torch.cumsum(variance, dim=0) / total_variance
    else:
        cumulative_variance_ratio = torch.zeros_like(variance)

    return U, S, cumulative_variance_ratio
