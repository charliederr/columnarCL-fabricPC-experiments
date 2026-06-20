# Depth-Spanning Columnar Architecture

## Problem Statement

The current ColBa implementation yields 10% accuracy on CIFAR-10 (random chance), compared to 42.45% for the PC ResNet baseline. The root cause: columns are implemented as a **single flat layer after the backbone**, not as structures that **span multiple depth levels** with skip connections.

## Architectural Goal

Redesign columns to span ResNet stages 2-4, with:
1. **K (Kernel)**: Receives concatenated features from all stages via skip connections
2. **L (Lateral)**: Refines local spatial structure from mid-level features
3. **B (Bridge)**: Broadcasts global context from the deepest stage

## Architecture Diagram

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                    COLUMN c                              │
                              │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
                              │  │     K      │  │     L      │  │     B      │          │
                              │  │  (kernel)  │  │  (lateral) │  │  (bridge)  │          │
                              │  │            │  │            │  │            │          │
     ┌───────────┐            │  │ ┌────────┐ │  │ ┌────────┐ │  │ ┌────────┐ │          │
     │ ResNet    │            │  │ │ K_deep │ │  │ │ L_deep │ │  │ │ B_deep │ │          │
     │ Stage 4   │───────────────▶│ (64→μd) │ │  │ │ (64→μd)│ │  │ │ (64→μd)│◀───global───│
     │ (8×8×64)  │            │  │ └────┬───┘ │  │ └────┬───┘ │  │ └────┬───┘ │  context  │
     └─────┬─────┘            │  │      │     │  │      │     │  │      │     │          │
           │                  │  │      ▼     │  │      ▼     │  │      ▼     │          │
           │ skip             │  │ ┌────────┐ │  │ ┌────────┐ │  │ ┌────────┐ │          │
           │                  │  │ │ K_mid  │ │  │ │ L_mid  │ │  │ │ B_mid  │ │          │
     ┌─────▼─────┐            │  │ │ +stage3│ │  │ │ 3×3    │ │  │ │        │ │          │
     │ ResNet    │───────────────▶│ │ concat │ │  │ │ conv   │ │  │ │        │ │          │
     │ Stage 3   │            │  │ └────┬───┘ │  │ └────┬───┘ │  │ └────┬───┘ │          │
     │ (16×16×32)│            │  │      │     │  │      │     │  │      │     │          │
     └─────┬─────┘            │  │      ▼     │  │      ▼     │  │      ▼     │          │
           │                  │  │ ┌────────┐ │  │ ┌────────┐ │  │ ┌────────┐ │          │
           │ skip             │  │ │ K_out  │ │  │ │ L_out  │ │  │ │ B_out  │ │          │
           │                  │  │ │ +stage2│ │  │ │ local  │ │  │ │ bcast  │ │          │
     ┌─────▼─────┐            │  │ │ concat │ │  │ │ refine │ │  │ │ expand │ │          │
     │ ResNet    │───────────────▶│ └────┬───┘ │  │ └────┬───┘ │  │ └────┬───┘ │          │
     │ Stage 2   │            │  │      │     │  │      │     │  │      │     │          │
     │ (32×32×16)│            │  │      ▼     │  │      ▼     │  │      ▼     │          │
     └─────┬─────┘            │  └──────┼─────┘  └──────┼─────┘  └──────┼─────┘          │
           │                  │         │              │              │                  │
           ▼                  │         └──────────────┼──────────────┘                  │
     ┌───────────┐            │                        ▼                                 │
     │  Input    │            │                  ┌───────────┐                           │
     │ (32×32×3) │            │                  │  K + L + B │  column output           │
     └───────────┘            │                  │  weighted  │  (tokens, embed_dim)     │
                              │                  └─────┬─────┘                           │
                              └────────────────────────┼─────────────────────────────────┘
                                                       │
                                                       ▼
                                          ┌─────────────────────────┐
                                          │   Column Combiner       │
                                          │   (sparse support)      │
                                          └───────────┬─────────────┘
                                                      │
                                                      ▼
                                          ┌─────────────────────────┐
                                          │      Classifier         │
                                          └─────────────────────────┘
```

## Symbol Table

| Symbol | Definition |
|--------|------------|
| `μd` | Microcolumn dimension (default 32), the internal width of each K/L/B pathway |
| `stage{2,3,4}` | ResNet stage outputs at progressively coarser spatial resolution |
| `K_deep/mid/out` | Kernel microcolumn layers processing stage4/3/2 features respectively |
| `L_deep/mid/out` | Lateral microcolumn layers with 3×3 conv for local refinement |
| `B_deep/out` | Bridge microcolumn: global pool → broadcast expansion |
| `tokens` | Number of spatial tokens (e.g., 64 = 8×8 grid from stage4) |
| `embed_dim` | Column output dimension per token |

## K/L/B Differentiation

| Microcolumn | Primary Input | Role | Mechanism |
|-------------|--------------|------|-----------|
| **K** (Kernel) | All stages via skip | Stable multi-scale | Each layer concatenates previous output with the corresponding stage features |
| **L** (Lateral) | Stage 4, refining downward | Local spatial detail | 3×3 depthwise-separable conv at each layer |
| **B** (Bridge) | Stage 4 global pool | Context broadcast | Single global vector broadcast to all tokens |

## Alternative Approaches Considered

### A. Keep columns flat, add more columns
- **Pros**: Minimal code change
- **Cons**: Does not address the fundamental depth-spanning requirement; adding capacity without fixing architecture

### B. Replace columns with transformer blocks
- **Pros**: Proven architecture, attention provides global context
- **Cons**: Loses the columnar specialization hypothesis; becomes standard ViT

### C. Depth-spanning columns with shared weights across stages (current proposal)
- **Pros**: Multi-scale features, skip connections for gradient flow, differentiates K/L/B roles
- **Cons**: More complex graph construction, larger model

### D. Separate backbone from columns entirely (late fusion)
- **Pros**: Simpler; backbone trains first, columns added later
- **Cons**: Columns cannot influence backbone via PC inference; misses joint optimization

**Chosen: Option C** — It most directly addresses the paper's insight that columns should span cortical layers (depth levels).

## Implementation Plan

### Phase 1: Stage-Tapping Infrastructure

Create utilities to extract features from multiple ResNet stages:

```python
# New file: columnar_cl_fabricpc/columns/stage_taps.py

class StageTapTokenizer(NodeBase):
    """
    Convert a stage's spatial features into tokens at a target resolution.

    Handles spatial mismatch by adaptive pooling/interpolation:
    - Stage 2 (32×32) → pool to (8×8) → 64 tokens
    - Stage 3 (16×16) → pool to (8×8) → 64 tokens
    - Stage 4 (8×8) → direct → 64 tokens
    """
    pass
```

### Phase 2: DepthSpanningColumn Node

Replace the flat `TypedColBaColumnNode` with a depth-spanning version:

```python
# New file: columnar_cl_fabricpc/columns/depth_spanning_column.py

class DepthSpanningColumnNode(NodeBase):
    """
    A column that spans multiple ResNet stages.

    Slots:
        - stage2: Input from ResNet stage 2 (tokenized)
        - stage3: Input from ResNet stage 3 (tokenized)
        - stage4: Input from ResNet stage 4 (tokenized)
        - stage4_pool: Global pooled features from stage 4

    Internal structure:
        K pathway: stage4 → K_deep → concat(stage3) → K_mid → concat(stage2) → K_out
        L pathway: stage4 → L_deep → 3×3 conv → L_mid → L_out
        B pathway: stage4_pool → B_deep → broadcast → B_out

    Output: weighted sum of K + L + B
    """

    @staticmethod
    def get_slots():
        return {
            "stage2": SlotSpec(name="stage2", is_multi_input=False),
            "stage3": SlotSpec(name="stage3", is_multi_input=False),
            "stage4": SlotSpec(name="stage4", is_multi_input=False),
            "stage4_pool": SlotSpec(name="stage4_pool", is_multi_input=False),
        }
```

### Phase 3: Graph Construction Refactor

Modify the training script to build the depth-spanning graph:

```python
# In train_cifar10_colba_accuracy.py

def build_depth_spanning_column_graph(
    num_columns: int,
    stage2_shape: Tuple[int, ...],  # (32, 32, 16)
    stage3_shape: Tuple[int, ...],  # (16, 16, 32)
    stage4_shape: Tuple[int, ...],  # (8, 8, 64)
    tokens: int = 64,
    embed_dim: int = 64,
    microcolumn_dim: int = 32,
) -> Tuple[List[NodeBase], List[Edge]]:
    """
    Build graph with depth-spanning columns.

    Each column receives skip connections from stages 2, 3, 4.
    """
    nodes = []
    edges = []

    # Stage tokenizers (shared across columns)
    stage2_tok = StageTapTokenizer(
        shape=(tokens, embed_dim),
        name="stage2_tok",
        source_shape=stage2_shape,
    )
    stage3_tok = StageTapTokenizer(...)
    stage4_tok = StageTapTokenizer(...)
    stage4_pool = GlobalPoolNode(shape=(1, embed_dim), name="stage4_pool")

    nodes.extend([stage2_tok, stage3_tok, stage4_tok, stage4_pool])

    # Per-column depth-spanning structure
    for c in range(num_columns):
        col = DepthSpanningColumnNode(
            shape=(tokens, embed_dim),
            name=f"col{c}",
            microcolumn_dim=microcolumn_dim,
        )
        nodes.append(col)
        edges.extend([
            Edge(stage2_tok, col.slot("stage2")),
            Edge(stage3_tok, col.slot("stage3")),
            Edge(stage4_tok, col.slot("stage4")),
            Edge(stage4_pool, col.slot("stage4_pool")),
        ])

    return nodes, edges
```

### Phase 4: Incremental Testing

1. **Baseline sanity check**: Verify PC ResNet alone still achieves ~42% (no columns)
2. **Single column**: One depth-spanning column, expect >10% (proves column isn't destroying signal)
3. **Multiple columns with sum combiner**: 4 columns, verify consistent training
4. **Full configuration**: 40 columns with attention combiner, shared/adaptive split

### Phase 5: Skip Connection Verification

Add diagnostic logging to verify gradient flow through skip connections:

```python
# In training loop
def log_gradient_flow(graph_state, params):
    """Log gradient magnitudes at each skip connection point."""
    for node_name, state in graph_state.items():
        if "K_mid" in node_name or "K_out" in node_name:
            grad_norm = jnp.linalg.norm(state.latent_grad)
            print(f"{node_name} grad norm: {grad_norm:.4f}")
```

## Expected Outcome

With depth-spanning columns:
- Each column receives rich multi-scale information
- Skip connections preserve gradient flow and low-level detail
- K/L/B microcolumns have genuinely different computational roles
- Test accuracy should exceed the 42.45% PC baseline, not collapse to 10%

## File Changes

| File | Change |
|------|--------|
| `columnar_cl_fabricpc/columns/stage_taps.py` | New: StageTapTokenizer, GlobalPoolNode |
| `columnar_cl_fabricpc/columns/depth_spanning_column.py` | New: DepthSpanningColumnNode |
| `columnar_cl_fabricpc/columns/__init__.py` | Export new nodes |
| `scripts/train_cifar10_colba_accuracy.py` | Refactor graph construction for depth-spanning |
| `scripts/train_cifar10_colba_accuracy.py` | Add `--depth-spanning` flag for new architecture |

## Success Criteria

1. **Minimum viable**: Single depth-spanning column achieves >15% test accuracy
2. **Target**: Full configuration achieves >45% test accuracy (exceeds PC baseline)
3. **Diagnostic**: Gradient norms at skip connection points are non-negligible (>0.01)
