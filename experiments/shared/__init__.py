"""Shared utilities for persona landscape experiments."""
from .trait_config import (
    PERSONAS, PERSONA_PROMPTS, TRAITS, TRAIT_PROMPTS,
    PERSONA_TRAIT_MATRIX, get_trait_prompt, get_persona_trait_matrix_tensor,
)
from .trait_vectors import (
    compute_trait_vectors_per_persona,
    compute_global_trait_vectors,
    cross_persona_cosine_similarity,
    compute_trait_residuals,
    project_onto_trait_basis,
)
from .multi_turn import (
    build_multi_turn_prompt,
    generate_multi_turn,
    collect_multi_turn_hidden,
)
from .utils import (
    cohens_d,
    cohens_d_multivariate,
    svd_analysis,
    effective_rank,
    plot_heatmap,
    plot_svd_spectrum,
    plot_pca_scatter,
    save_run_config,
)
