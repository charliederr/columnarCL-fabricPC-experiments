# Codex Architecture Review, 2026-07-01

This file describes the current plain CIFAR-10 depth-spanning columnar predictive-coding implementation in `columnarCL-fabricPC-experiments`. It is an architecture review, not a new experiment plan.

The current code uses the upstream FabricPC checkout at `/home/ni/repos/fpc/FabricPC` through the virtual environment `/home/ni/repos/fpc/virt-envs/fpcpy3.12`. The experiment repository adds local nodes, graph construction, shell mechanics, readout ablations, and run scripts without modifying FabricPC.

## Scope

The current tested graph is built by `scripts/train_cifar10_depth_spanning.py` with the recent shell dynamics settings:

- `--model resnet18`
- `--num_columns 4`
- `--column_mode all_active`
- `--combiner sum`
- `--embed_dim 64`
- `--microcolumn_dim 32`
- `--layer_norm_tokens`
- `--fix_ln_gamma`
- `--column_teacher_weight 0.0`
- `--shell_teacher_weights 0,0,0,0`
- `--column_shell_teacher_weights 0,0,0,0`
- `--column_shell_readout`
- `--column_shell_bridge`
- `--shell_evidence_cascade_scale 0.05,0.05,0.05`
- `--shell_inhibition_strengths 0,0.35,0.22,0.10` for the full-inhibition run

For this configuration, `build_depth_spanning_graph(args)` constructs 77 nodes and 131 edges. The active support mask is `(1.0, 1.0, 1.0, 1.0)`, so all four columns are active.

## Symbol Table

`x` is the CIFAR-10 image tensor clamped to the FabricPC source node named `input`.

`y` is the one-hot CIFAR-10 class target tensor clamped to the main classifier node named `output` during training.

`z_latent` is the inferred latent state stored in each FabricPC `NodeState`.

`z_mu` is the prediction that a node computes for its own latent state from its input nodes and parameters.

`E` is a node energy value computed from `z_latent` and `z_mu`.

`K` is the kernel pathway inside `DepthSpanningColumnNode`, implemented as a multi-stage path that combines `stage4`, `stage3`, and `stage2` tokens.

`L` is the lateral pathway inside `DepthSpanningColumnNode`, implemented as a local depthwise convolution over the token grid.

`B` is the bridge pathway inside `DepthSpanningColumnNode`, implemented as a global `stage4_pool` context vector projected and broadcast to all tokens.

`c` is a column index in `col_00`, `col_01`, `col_02`, or `col_03`.

`support_mask` is the fixed vector used by `MaskedColumnCombinerNode` to select which columns contribute to the combiner output.

`hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell` are the four ordered slices of each column's output feature axis.

`shell_path_scale` is the learnable 4 by 3 matrix inside each depth-spanning column. Its rows correspond to the four shell slices and its columns correspond to `K`, `L`, and `B`.

## Code Map

Current experiment entry point:

- `scripts/train_cifar10_depth_spanning.py`: constructs the ResNet-like predictive-coding graph, adds depth-spanning columns, adds shell readout and bridge paths, trains with FabricPC, and reports ablations.

Local architecture nodes:

- `columnar_cl_fabricpc/columns/depth_spanning_column.py`: implements `DepthSpanningColumnNode`, shell slices, K/L/B pathways, same-tier inhibition, outward shell evidence cascade, and optional shell-wise LayerNorm.
- `columnar_cl_fabricpc/columns/stage_taps.py`: implements `StageTapTokenizer` and `GlobalPoolNode`.
- `columnar_cl_fabricpc/columns/accuracy_nodes.py`: implements `MaskedColumnCombinerNode` and `FeatureSliceNode`. It also contains older accuracy-oriented nodes that are not used by the current depth-spanning script.
- `columnar_cl_fabricpc/columns/label_smoothed_ce.py`: implements `WeightedLabelSmoothedCrossEntropyEnergy`.

Upstream FabricPC mechanics used by the experiment:

- `/home/ni/repos/fpc/FabricPC/fabricpc/graph_assembly/graph_construction.py`: builds `GraphStructure`, validates slots, resolves `TaskMap`, and attaches muPC scaling.
- `/home/ni/repos/fpc/FabricPC/fabricpc/core/types.py`: defines `GraphStructure`, `GraphParams`, `GraphState`, `NodeInfo`, `NodeState`, and `EdgeInfo`.
- `/home/ni/repos/fpc/FabricPC/fabricpc/nodes/base.py`: defines the node contract and the autodiff paths for latent gradients and local weight gradients.
- `/home/ni/repos/fpc/FabricPC/fabricpc/core/inference.py`: defines the predictive-coding inference loop and `InferenceSGDNormClip`.
- `/home/ni/repos/fpc/FabricPC/fabricpc/core/learning.py`: computes local node parameter gradients after inference.
- `/home/ni/repos/fpc/FabricPC/fabricpc/training/train.py`: runs training and evaluation loops.

## Current Graph Overview

The current graph starts with a predictive-coding ResNet-like visual backbone, taps the final three stages into common token tensors, sends those tokens to every active depth-spanning column, and reads out CIFAR-10 classes through several column paths.

```text
input image x
    |
    v
stem ConvNode
    |
    v
ResNet-style predictive-coding backbone
    |
    +--> stage2_out ----> stage2_tap ----+
    |                                     |
    +--> stage3_out ----> stage3_tap ----+--> col_00 --+
    |                                     |             |
    +--> stage4_out ----> stage4_tap ----+--> col_01 --+
    |                                     |             |
    +--> stage4_out ----> stage4_pool ---+--> col_02 --+--> combiner --> column_pool --> output
                                          |             |                         |          ^
                                          +--> col_03 --+                         |          |
                                                                                v          |
                                                                    column_teacher_output   |
                                                                                           |
per-column shell readout and bridge paths ------------------------------------------------+
```

The main classifier is `output`. In recent runs without `--bypass_columns`, the main classifier receives:

- `column_pool -> output`
- each active per-column shell pool directly into `output`
- each active per-column shell bridge into `output`

The column teacher node exists in the graph as `column_teacher_output`, but recent shell dynamics runs set `--column_teacher_weight 0.0`. That means its weighted cross-entropy energy is zero in those runs.

## ResNet-Style Backbone

The backbone is constructed by `make_residual_block()` and `build_depth_spanning_graph()` in `scripts/train_cifar10_depth_spanning.py`.

For `resnet18`, the model configuration is:

```text
stem_channels = 32
stage 1: channels=32,  blocks=2, first_stride=1
stage 2: channels=64,  blocks=2, first_stride=2
stage 3: channels=128, blocks=2, first_stride=2
stage 4: channels=256, blocks=2, first_stride=2
```

The backbone node pattern is:

```text
prev
 |\
 | \ if shape changes or stride > 1
 |  \
 |   v
 |  skip projection ConvNode
 |
 v
conv_a ConvNode
 |
 v
conv_b ConvNode
 |
 v
skip_sum SkipConnection
 ^
 |
prev or skip projection
```

`ConvNode` comes from upstream FabricPC. It computes a convolutional `pre_activation`, applies the configured activation, computes `error = z_latent - z_mu`, and stores Gaussian energy by default. Its fan-in for muPC scaling is `input_channels * product(kernel_size)`.

`SkipConnection` comes from upstream FabricPC. It sums its inputs without learnable parameters and marks its input slot as not variance-scalable so muPC does not attenuate identity residual paths.

The current depth-spanning columns use the last three backbone stage outputs:

```text
stage2_out = stage_outputs[-3] = (16, 16, 64)
stage3_out = stage_outputs[-2] = (8, 8, 128)
stage4_out = stage_outputs[-1] = (4, 4, 256)
target_grid = (4, 4)
num_tokens = 16
```

These values are printed by `build_depth_spanning_graph()` when the graph is built.

## Stage Taps

`StageTapTokenizer` converts a spatial backbone tensor into a common token tensor. In code, the transformation is:

```text
(batch, height, width, channels)
    -> adaptive average pool to target_grid
    -> reshape to (batch, tokens, channels)
    -> multiply by W_proj and add b_proj
    -> add pos_embed if enabled
    -> optionally apply LayerNorm
    -> z_mu for the stage tap node
```

The implementation is in `columnar_cl_fabricpc/columns/stage_taps.py`.

The current graph creates:

```text
stage2_out (16, 16, 64)  -> stage2_tap (16, 64)
stage3_out (8, 8, 128)  -> stage3_tap (16, 64)
stage4_out (4, 4, 256)  -> stage4_tap (16, 64)
stage4_out (4, 4, 256)  -> stage4_pool (1, 64)
```

`GlobalPoolNode` computes `stage4_pool`. Its transformation is:

```text
(batch, height, width, channels)
    -> mean over height and width
    -> multiply by W_proj and add b_proj
    -> expand to (batch, 1, embed_dim)
    -> optionally apply LayerNorm
    -> z_mu for stage4_pool
```

In recent runs, `--layer_norm_tokens --fix_ln_gamma` makes each stage tap and `stage4_pool` use scalar gamma `1.0` and beta `0.0`, not learned LayerNorm gain and bias.

## Depth-Spanning Column Inputs

Each column has four single-input slots:

```text
stage2      receives stage2_tap
stage3      receives stage3_tap
stage4      receives stage4_tap
stage4_pool receives stage4_pool
```

The graph repeats this edge pattern for every column:

```text
stage2_tap  -> col_XX:stage2
stage3_tap  -> col_XX:stage3
stage4_tap  -> col_XX:stage4
stage4_pool -> col_XX:stage4_pool
```

The relevant code is `build_depth_spanning_graph()` in `scripts/train_cifar10_depth_spanning.py`, where `create_depth_spanning_column()` is called for each `idx` and the four edges are added.

## Inside One Depth-Spanning Column

Each column maps the four inputs to `(batch, tokens, embed_dim)`. For recent runs, this is `(batch, 16, 64)`.

The internal computation is:

```text
                 stage4_tap
                    |
                    +--------------------+
                    |                    |
                    v                    v
             K_deep path             L_deep path
                    |                    |
stage3_tap -------->+                    v
                    |              depthwise 3x3 conv
                    v                    |
              K_mid path                 v
                    |                 L_out
stage2_tap -------->+
                    |
                    v
                 K_out

stage4_pool
    |
    v
B_deep -> B_out -> broadcast over all tokens

K_out, L_out, and B_out each have shape (batch, tokens, embed_dim).
They are sliced into hard_kernel, inner_shell, middle_shell, and outer_shell.
```

The K, L, and B paths are implemented in `DepthSpanningColumnNode.forward()`:

- `K`: `stage4 -> K_W_deep -> concat(stage3) -> K_W_mid -> concat(stage2) -> K_W_out`.
- `L`: `stage4 -> L_W_deep -> reshape to grid -> 3 by 3 depthwise convolution -> L_W_out`.
- `B`: `stage4_pool -> B_W_deep -> B_W_out -> broadcast to every token`.

The learnable parameters are initialized in `DepthSpanningColumnNode.initialize_params()`:

```text
K_W_deep: input_dim -> microcolumn_dim
K_W_mid:  microcolumn_dim + input_dim -> microcolumn_dim
K_W_out:  microcolumn_dim + input_dim -> output_dim

L_W_deep:    input_dim -> microcolumn_dim
L_W_conv_dw: 3 by 3 depthwise convolution over microcolumn_dim channels
L_W_out:     microcolumn_dim -> output_dim

B_W_deep: input_dim -> microcolumn_dim
B_W_out:  microcolumn_dim -> output_dim
```

## Column Shell Layout

The column output feature axis is partitioned into four named slices:

```text
feature axis, embed_dim = 64

0                                                           64
|----------------------|-------|--------------|-------------|
 hard_kernel            inner   middle_shell   outer_shell
 width 22               width 7 width 14       width 21
```

The default proportions are `32:10:20:30`. For `embed_dim = 64`, `compute_shell_sizes()` produces widths `22, 7, 14, 21`.

The shell order is fixed in `SHELL_NAMES`:

```text
hard_kernel -> inner_shell -> middle_shell -> outer_shell
```

This ordering matters for feature slicing, same-tier inhibition, shell readout nodes, and outward shell evidence cascade.

## Shell-To-Path Mask

Each shell receives a fixed subset of K/L/B pathway evidence. The active subset is encoded in `DEFAULT_SHELL_PATH_MASK`.

```text
                 K        L        B
hard_kernel      1        0        0
inner_shell      1        1        0
middle_shell     1        1        1
outer_shell      0        1        1
```

The learnable `shell_path_scale` matrix has the same shape. The forward pass multiplies it by the fixed mask before combining K/L/B slices. A masked entry stays zero in the forward computation even though `shell_path_scale` itself is a parameter tensor.

For one shell slice, the computation is:

```text
shell_output =
    shell_path_scale[shell, K] * K_out_shell
  + shell_path_scale[shell, L] * L_out_shell
  + shell_path_scale[shell, B] * B_out_shell
```

The hard kernel receives only K pathway output. The outer shell receives only L and B pathway output.

## Same-Tier Inhibition

Same-tier inhibition is implemented in `_same_tier_inhibition()` and `_apply_pathway_shell_inhibition()` in `columnar_cl_fabricpc/columns/depth_spanning_column.py`.

It is applied separately to each shell slice inside each of K, L, and B before shell-path mixing:

```text
K_out -> split by shell -> inhibit each shell -> concatenate
L_out -> split by shell -> inhibit each shell -> concatenate
B_out -> split by shell -> inhibit each shell -> concatenate
```

For one shell tier, the operation is:

```text
magnitude = abs(shell_output)
other_mean = (sum(magnitude over shell features) - magnitude) / (width - 1)
inhibited_magnitude = relu(magnitude - strength * other_mean)
inhibited_shell_output = sign(shell_output) * inhibited_magnitude
```

`strength` is the inhibition coefficient for that shell tier. The full-inhibition run used:

```text
hard_kernel  = 0.00
inner_shell  = 0.35
middle_shell = 0.22
outer_shell  = 0.10
```

The half-inhibition diagnostic used:

```text
hard_kernel  = 0.000
inner_shell  = 0.175
middle_shell = 0.110
outer_shell  = 0.050
```

The hard kernel is not inhibited in either setting.

## Outward Shell Evidence Cascade

The current code implements an outward shell-to-shell evidence path. It is not HiBaCaML consolidation.

The current cascade direction is:

```text
hard_kernel -> inner_shell -> middle_shell -> outer_shell
```

For each adjacent pair, there is a learned matrix and bias:

```text
shell_evidence_cascade_hard_kernel_to_inner_shell
shell_evidence_cascade_inner_shell_to_middle_shell
shell_evidence_cascade_middle_shell_to_outer_shell
```

The forward pass adds:

```text
target_shell = target_shell + scale[pair] * (source_shell @ W_pair + b_pair)
```

Recent runs use `scale = 0.05,0.05,0.05`.

This path is useful as shell communication, but it is opposite to the consolidation mechanism described in the paper. The paper's consolidation moves reusable material inward, while this implemented path sends evidence outward.

## Column LayerNorm

If `--layer_norm_tokens` is set, each shell output is normalized independently before the shell outputs are concatenated. This is implemented in `_shellwise_layernorm()`.

If `--fix_ln_gamma` is also set, gamma is scalar `1.0` and beta is scalar `0.0`. The recent runs use both flags. That removes the learned LayerNorm rescale path in the stage taps, `stage4_pool`, and depth-spanning columns.

## Column Output As A Predictive-Coding Node

After K/L/B mixing, same-tier inhibition, outward cascade, and optional shell-wise LayerNorm, the column concatenates shell outputs:

```text
pre_activation = concat(
    hard_kernel_output,
    inner_shell_output,
    middle_shell_output,
    outer_shell_output
)
z_mu = activation(pre_activation)
error = z_latent - z_mu
E = GaussianEnergy(z_latent, z_mu)
```

The column is still a FabricPC node. Its latent `z_latent` is inferred during predictive-coding inference. Its parameters are updated by local gradients after inference.

## Combiner

`MaskedColumnCombinerNode` receives every column output, applies `support_mask`, and produces `(batch, tokens, embed_dim)`.

For `--combiner sum`, the current recent setting, the computation is:

```text
pre_activation = sum_i support_mask[i] * col_i
pre_activation = pre_activation / sqrt(active_count)
```

For the current active mask, `active_count = 4`. The combiner divides the summed columns by `2.0`.

For `--combiner attention`, which is implemented but not used in recent runs, the node learns `col_attention`, masks inactive columns with a large negative logit, applies softmax, and returns the weighted sum.

## Main Readout And Auxiliary Readouts

The main readout path is:

```text
col_00 \
col_01  \
col_02   +--> combiner -> column_pool -> output
col_03  /
```

`column_pool` is upstream FabricPC `AvgPool(global_pool=True)`. It averages the combiner over the token axis and outputs `(batch, embed_dim)`.

`output` is upstream FabricPC `Linear(shape=(10,), activation=SoftmaxActivation(), flatten_input=True)`. It uses `WeightedLabelSmoothedCrossEntropyEnergy` with weight `1.0` and the configured label smoothing value.

The column teacher path is always created:

```text
combiner -> column_pool -> column_teacher_output
```

`column_teacher_output` is also a softmax `Linear(shape=(10,), flatten_input=True)`. Recent shell dynamics runs set `--column_teacher_weight 0.0`, so this node's energy does not contribute class error in those runs.

## Per-Column Shell Direct Readout

When `--column_shell_readout` is set, each active column also exposes each shell slice directly to the classifier.

For one column, the pattern is:

```text
col_c
  |
  +--> columnCC_hard_kernel_slice  -> columnCC_hard_kernel_pool  -> output
  +--> columnCC_inner_shell_slice  -> columnCC_inner_shell_pool  -> output
  +--> columnCC_middle_shell_slice -> columnCC_middle_shell_pool -> output
  +--> columnCC_outer_shell_slice  -> columnCC_outer_shell_pool  -> output
```

`FeatureSliceNode` performs the contiguous feature-axis slice and has no learnable parameters. `AvgPool(global_pool=True)` pools the token axis. The direct shell readout preserves column identity and shell identity until the final classifier node.

For the current graph with four active columns, this creates 16 shell slice nodes and 16 shell pool nodes.

## Per-Column Shell Bridge

When `--column_shell_bridge` is set, each active column has one bridge node receiving all four pooled shell vectors.

For one column, the pattern is:

```text
columnCC_hard_kernel_pool  \
columnCC_inner_shell_pool   \
columnCC_middle_shell_pool   +--> columnCC_shell_bridge -> output
columnCC_outer_shell_pool   /
```

`columnCC_shell_bridge` is an upstream FabricPC `Linear(shape=(embed_dim,), activation=IdentityActivation(), flatten_input=False)`. Because `flatten_input=False`, each input edge is transformed along its last feature axis into `embed_dim`, and the transformed shell inputs are summed.

This bridge is a predictive-coding latent with Gaussian energy. It is not just a readout-time concatenation. During inference, its latent participates in the graph energy and can send gradient contributions back to its shell-pool inputs.

## Optional Bypass Path

The optional bypass path is controlled by `--bypass_columns`:

```text
stage4_out -> bypass_pool -> output
```

Recent shell dynamics runs did not use the bypass. Earlier experiments used it to separate direct backbone evidence from column-mediated evidence. The current graph builder still supports it.

## Optional Shell Teacher Heads

The graph builder can create shell-local teacher heads from `column_pool` when `--shell_teacher_weights` contains positive weights:

```text
column_pool -> hard_kernel_slice  -> hard_kernel_teacher_output
column_pool -> inner_shell_slice  -> inner_shell_teacher_output
column_pool -> middle_shell_slice -> middle_shell_teacher_output
column_pool -> outer_shell_slice  -> outer_shell_teacher_output
```

Recent shell dynamics runs set `--shell_teacher_weights 0,0,0,0`, so these nodes were omitted.

The graph builder can also create per-column shell teacher heads when `--column_shell_teacher_weights` contains positive weights:

```text
columnCC_shell_pool -> columnCC_shell_teacher_output
```

Recent shell dynamics runs set `--column_shell_teacher_weights 0,0,0,0`, so these teacher classifier nodes were omitted. The per-column shell pools still existed because direct shell readout and shell bridge needed them.

## Current Tested Node Families

For the current full-inhibition graph with four active columns, no bypass, direct shell readout on, and shell bridge on, the node families are:

```text
input source:
  input

backbone:
  stem
  s1b1_conv_a, s1b1_conv_b, s1b1_skip_sum
  s1b2_conv_a, s1b2_conv_b, s1b2_skip_sum
  s2b1_conv_a, s2b1_conv_b, s2b1_skip_sum, s2b1_skip_proj
  s2b2_conv_a, s2b2_conv_b, s2b2_skip_sum
  s3b1_conv_a, s3b1_conv_b, s3b1_skip_sum, s3b1_skip_proj
  s3b2_conv_a, s3b2_conv_b, s3b2_skip_sum
  s4b1_conv_a, s4b1_conv_b, s4b1_skip_sum, s4b1_skip_proj
  s4b2_conv_a, s4b2_conv_b, s4b2_skip_sum

stage taps:
  stage2_tap
  stage3_tap
  stage4_tap
  stage4_pool

columns:
  col_00
  col_01
  col_02
  col_03

combiner and pooled readout:
  combiner
  column_pool
  output
  column_teacher_output

per-column shell nodes, repeated for each active column:
  columnCC_hard_kernel_slice
  columnCC_hard_kernel_pool
  columnCC_inner_shell_slice
  columnCC_inner_shell_pool
  columnCC_middle_shell_slice
  columnCC_middle_shell_pool
  columnCC_outer_shell_slice
  columnCC_outer_shell_pool
  columnCC_shell_bridge
```

This totals 77 nodes.

## Predictive-Coding Mechanics

Every node follows the FabricPC `NodeBase.forward()` contract:

```text
inputs and params -> pre_activation -> z_mu
error = z_latent - z_mu
E = energy(z_latent, z_mu)
```

The universal node state fields are defined by FabricPC `NodeState`:

```text
z_latent
z_mu
error
energy
pre_activation
latent_grad
```

The graph is static. `GraphStructure` stores nodes, edges, task map, topological order, and config. `GraphParams` stores node-local weights and biases.

### Training Clamps

During training, FabricPC maps batch dictionary keys to nodes through `TaskMap`.

The current depth-spanning graph always maps:

```text
x        -> input
y        -> output
column_y -> column_teacher_output
```

`ColumnTeacherTargetLoader` duplicates the one-hot CIFAR-10 label from `y` into `column_y`. If shell teacher heads are active, it also duplicates `y` into the shell target keys.

This means during training:

```text
input.z_latent                 is clamped to x
output.z_latent                is clamped to y
column_teacher_output.z_latent is clamped to column_y
```

If a teacher energy weight is zero, the node can still be clamped but its weighted cross-entropy energy contributes zero.

### State Initialization

FabricPC graph construction defaults to `FeedforwardStateInit()` when no custom initializer is supplied. The graph builder sets this default in `graph_construction.py`.

`FeedforwardStateInit` first initializes all `z_latent` tensors from their node latent initializers and overlays clamps. It then walks the graph in topological order. For non-clamped internal nodes, it computes `z_mu` and sets `z_latent = z_mu` before inference begins.

### Inference Loop

The current graph uses `InferenceSGDNormClip`:

```text
eta_infer = args.eta_infer
infer_steps = args.infer_steps
max_norm = args.infer_max_norm
```

Recent runs used:

```text
eta_infer = 0.1
infer_steps = 40
max_norm = 1.0
```

For each inference step, FabricPC does:

```text
1. zero every node's latent_grad
2. run nodes in graph order
3. for each node, compute z_mu, error, energy
4. use autodiff to compute:
     dE/d(node z_latent)
     dE/d(each input edge latent)
5. accumulate latent_grad into the node and its input source nodes
6. update every unclamped z_latent:
     grad_norm = L2 norm over non-batch axes
     clipped_grad = grad * min(1, max_norm / (grad_norm + eps))
     z_latent = z_latent - eta_infer * clipped_grad
```

Clamped nodes are not updated in the inference step.

### Local Parameter Learning

After inference, `get_graph_param_gradient()` computes local weight gradients with `compute_local_weight_gradients()`.

For each node, FabricPC gathers the final inferred input latents, runs the node's own forward computation, and uses `jax.value_and_grad()` on that node's energy with respect to that node's parameters. This is local in the sense that the parameter gradient for one node is computed from that node's final inputs, latent, prediction, and energy. The optimizer then applies those gradients through Optax AdamW.

The current training script uses:

```text
optax.warmup_cosine_decay_schedule
optax.adamw
weight_decay = args.weight_decay
```

Recent runs used `lr = 0.005` and `weight_decay = 0.01`.

### Evaluation

During evaluation, FabricPC clamps only `x` to `input`. It does not clamp `y` to `output`. The output node is terminal and unclamped, so FabricPC sets `output.z_latent = output.z_mu` and zeroes its energy. Accuracy is computed from `argmax(output.z_mu)` against the batch labels.

This is why evaluation energy is not the classifier loss. Evaluation accuracy is the relevant metric for CIFAR classification.

## Ablation Mechanisms

The ablation helpers do not change the graph topology. They copy trained parameters and zero selected classifier or bridge weights before evaluation.

Readout source ablations:

- `evaluate_readout_ablations()` masks input edges into `output`.
- `column_only` keeps `column_pool -> output`.
- `column_shell_readout_only` keeps direct shell pool edges into `output`.
- `column_shell_bridge_only` keeps shell bridge edges into `output`.
- `column_shell_readout_plus_bridge` keeps both direct shell pools and shell bridges.

Shell path ablations:

- `evaluate_column_shell_readout_ablations()` isolates direct pooled shell evidence.
- `evaluate_column_shell_bridge_ablations()` isolates the shell bridge path and masks bridge inputs by shell.
- `evaluate_column_shell_path_ablations()` combines direct shell readout and shell bridge paths, then keeps or drops one shell in both routes.

These ablations explain the recent conclusion that `outer_shell` is not independently class-readable but can contribute contextually through the combined shell path.

## What Has Been Implemented From HiBaCaML And ColBa

The current code implements these structural pieces:

1. Structurally restricted component learners.

   The depth-spanning columns are restricted components. Each column has a fixed K/L/B internal pathway structure, a fixed shell order, and fixed shell-to-path masks.

2. Typed microcolumn pathways.

   K, L, and B exist as distinct computations inside `DepthSpanningColumnNode`. K integrates multiple backbone stages. L mixes local token neighborhoods. B broadcasts global context.

3. Hard kernel plus shells.

   Each column output is partitioned into `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell` slices. The hard kernel is a normal learned feature slice, but it is protected from same-tier inhibition by the default inhibition coefficient of zero.

4. Same-tier inhibition.

   Same-tier feature competition is implemented separately inside K, L, and B shell slices before shell mixing.

5. Sparse support representation at graph construction time.

   `support_mask` allows all-active, first-sparse, or random-sparse column support. The topology keeps all columns wired, and `MaskedColumnCombinerNode` masks inactive columns.

6. Combiner.

   `MaskedColumnCombinerNode` implements sum and attention modes over the active columns.

7. Per-column shell readout.

   The current graph can preserve column and shell identity until `output` through direct per-column shell pools.

8. Per-column shell bridge.

   The current graph can route all shell pools for one column through a predictive-coding latent before `output`.

9. Auxiliary teacher heads as graph nodes.

   The code can attach a column teacher head, shell teacher heads, and per-column shell teacher heads. Recent runs set their weights to zero, but the machinery exists.

10. Shell diagnostics.

   The code reports shell norms and evaluates shell lesions without changing the trained graph.

## What Has Not Been Implemented From HiBaCaML And ColBa

The following paper mechanisms are not implemented in the current plain CIFAR-10 script.

1. A top-level probabilistic controller named `Psi`.

   The paper describes a controller that selects sparse supports under uncertainty. The current code has `support_mask`, but it is fixed before training by command-line options. There is no learned or audited support posterior.

2. Teacher-first exact support search.

   The paper's ColBa regime evaluates candidate supports and applies exact or one-swap teacher corrections. The current code does not evaluate alternate support sets during training. It does not apply one-swap support changes.

3. Adaptive shared, non-shared, and reserve column pools.

   The paper describes always-on shared columns, adaptive columns, and reserve columns. The current code can create a fixed number of columns and choose a fixed support mask, but it does not reserve columns for later tasks or dynamically assign roles.

4. Internal probabilistic state per column.

   The paper uses internal probabilistic states that distinguish shared abstraction, semi-general structure, task-local residue, and stale material. The current code has deterministic feature slices and predictive-coding latents, but no explicit per-column probabilistic state variable representing those roles.

5. Upward internal certificates.

   The paper lists certificates such as shared-abstraction mass, specificity load, saturation pressure, demotion pressure, and similarity signatures. The current code logs diagnostics, but it does not produce certificate vectors that feed a controller.

6. Certificate-based support scoring.

   The paper's Rao-Blackwell argument applies when internal certificates improve reuse-utility estimates beyond external fit scores. The current code does not use certificates to select columns.

7. Outside-in pruning.

   The paper describes conservative pruning that starts in outer shells, then middle shells, with inner shells protected initially. The current code does not prune shell weights or feature units.

8. Inward promotion as consolidation.

   The paper describes promotion from outer shells toward inner shells when material becomes reusable. The current code implements the opposite direction as an outward evidence cascade from hard kernel to outer shell. That cascade is not consolidation.

9. Outward demotion as controlled forgetting.

   The paper describes demoting stale or over-specialized inner material outward. The current code has no demotion operation, no stale-feature detector, and no shell swap teacher.

10. Internal demotion-swap teacher.

   The paper describes a teacher that audits proposed internal shell swaps. The current code has evaluation-time ablations, but no training-time internal reorganization audit.

11. Multi-task or old-task retention objective.

   The current script trains plain CIFAR-10 as one 10-class task. It does not implement Split-CIFAR tasks, old-task heads, worst-old-task retention, task-boundary audits, or switching penalties.

12. Larger sparse CIFAR-scale column pool.

   The paper's CIFAR direction discusses larger pools with sparse active support. The current tested graph uses four active columns, all active.

13. Explicit non-adjacent depth routes beyond the current K skips.

   The current K path receives stage4, stage3, and stage2 through sequential concatenations. There are no separate named route families from shallow stages to deep shell states, no top-down prediction routes from deep shells to shallow stages, and no independently lesionable depth-route nodes.

14. Local task heads for split continual learning.

   The current `output` is one 10-way CIFAR-10 head. The split-task local head setup described for continual learning is not present in this script.

15. Transformer or reinforcement-learning extensions.

   The paper discusses possible extensions beyond vision classification. None of those are part of the current CIFAR-10 implementation.

## Current Architectural Facts To Keep In Mind

The columns are predictive-coding nodes, not ordinary feedforward-only modules. Their latents are inferred for each batch before local parameter gradients are computed.

The current shell bridge is also a predictive-coding latent. It can send inference gradients back to shell pools because it is a graph node, not only a classifier input.

The current per-column shell direct readout preserves shell and column identity until the classifier. The `column_pool` path does not preserve that identity because it averages after the combiner.

The current `outer_shell` can be useful only through coupling. In recent ablations, `outer_shell_only` stayed near chance, while removing `outer_shell` from the combined shell path damaged accuracy in several runs.

The current outward shell evidence cascade should not be described as faithful HiBaCaML promotion. It is a learned communication path in the opposite direction.

The current implementation has shell slices inside each K/L/B output before mixing, but it does not give each microcolumn an independent persistent hard-kernel-plus-shell state with promotion, demotion, and certificates.

The current support mask gives a static architectural support. It is not the paper's teacher-first support controller.

## Reference Pointers

Key code locations:

- `scripts/train_cifar10_depth_spanning.py`, `build_depth_spanning_graph()`: graph construction, backbone, stage taps, columns, combiner, readout paths, teacher heads, optional bypass, task map, and inference config.
- `scripts/train_cifar10_depth_spanning.py`, `train_cifar10_depth_spanning()`: data loaders, optimizer schedule, training loop, best-validation checkpoint selection, and ablation reporting.
- `columnar_cl_fabricpc/columns/depth_spanning_column.py`, `SHELL_NAMES`, `DEFAULT_SHELL_PATH_MASK`, and `DepthSpanningColumnNode`: shell layout and column internals.
- `columnar_cl_fabricpc/columns/depth_spanning_column.py`, `_same_tier_inhibition()`: same-tier inhibition operation.
- `columnar_cl_fabricpc/columns/stage_taps.py`, `StageTapTokenizer`: conversion from backbone stages to token streams.
- `columnar_cl_fabricpc/columns/stage_taps.py`, `GlobalPoolNode`: global context vector for B.
- `columnar_cl_fabricpc/columns/accuracy_nodes.py`, `MaskedColumnCombinerNode`: static support mask and active-column combination.
- `columnar_cl_fabricpc/columns/accuracy_nodes.py`, `FeatureSliceNode`: shell slicing for readout and diagnostics.
- `columnar_cl_fabricpc/columns/label_smoothed_ce.py`, `WeightedLabelSmoothedCrossEntropyEnergy`: weighted class energy for main and auxiliary classifier heads.
- `/home/ni/repos/fpc/FabricPC/fabricpc/nodes/base.py`, `NodeBase.forward_and_latent_grads()` and `NodeBase.forward_and_weight_grads()`: autodiff implementation of latent gradients and local weight gradients.
- `/home/ni/repos/fpc/FabricPC/fabricpc/core/inference.py`, `InferenceSGDNormClip`: clipped latent inference update.
- `/home/ni/repos/fpc/FabricPC/fabricpc/training/train.py`, `get_graph_param_gradient()` and `train_step()`: inference to convergence followed by local parameter gradients and optimizer update.
