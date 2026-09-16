# Day 3: Large-Scale Models on Alps

Status: living design document; decisions, assumptions, and open questions will change as the course is piloted.

Last updated: 2026-09-16

## Course context

Provisional three-day arc:

- Day 1: single-GPU performance.
- Day 2: distributed training with DDP and FSDP.
- Day 3: model parallelism, conditional computation, and end-to-end optimization.

The intended Day 3 audience is a mixed technical cohort. Participants should have completed Days 1 and 2, but their ML, PyTorch, and HPC experience will vary.

The primary outcome is not memorizing distributed APIs. By the end of Day 3, participants should be able to choose a plausible parallelization strategy for a model and Alps allocation, measure it, diagnose the limiting resource, and defend the decision with evidence.

## Current decisions

- Use MoE as the narrative connecting model architecture, parallelism, communication, numerical stability, and kernels.
- Preserve space for participant-selected detours.
- Triton is the primary hands-on custom-kernel environment.
- Use the existing working Mojo uenv for a comparison demo and, if time permits, an optional extension exercise.
- Clariden resources will be a shared pool rather than a fixed multi-node allocation per team.
- Agents are allowed and should be treated as part of current engineering practice.
- Agent-produced code or explanations are hypotheses, not evidence. Correctness checks and measurements on Alps remain required.
- Use a single configurable transformer/MoE workload throughout the course where practical.
- Use a pre-tokenized, source-labelled Dolma subset as the primary corpus; keep TinyStories as a lightweight fallback/development configuration.
- Provide pretrained semantic checkpoints and separate scaled systems configurations built from the same model and data interfaces.
- Build structured metrics, bounded multi-rank profiling, evaluation, and interpretation hooks into the shared harness before adding parallel strategies.
- Build the complete three-day replacement under a new top-level `course/` directory; keep the current materials intact and port selectively through dedicated Git move/copy commits so existing authorship remains inspectable.
- Use `jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/pytorch-cuda:26.07-py3-alps7-dev-14ac42497583f60a1dd156230f00af8f` as the canonical PyTorch runtime, pinned additionally by immutable digest.
- Follow the full-course rebuild plan in [`course-rebuild-plan.md`](course-rebuild-plan.md).

## Alps hardware model

These notes describe the hardware behavior relevant to course design. They should be validated and refined during course preparation, but should not be replaced with a generic DGX/NVSwitch mental model.

### Node structure and interconnects

- A Clariden GH200 node contains four Grace-Hopper modules.
- The H100 GPUs within a node have NVLink connectivity.
- There is no NVSwitch.
- There is C2C connectivity between the Grace chips.
- Within a Grace-Hopper module there is a fast Grace/Hopper interconnect.
- Each module is associated with a Slingshot network interface; inter-node communication uses the Alps Slingshot fabric.

The absence of NVSwitch does not mean that the GPUs have no fast intra-node connectivity. It means that topology and routing differ from an NVSwitch-based system and must be represented accurately in diagrams and performance reasoning.

### Unified, cache-coherent memory behavior

The memory of the node, including GPU HBM, can appear as cache-coherent unified memory when allocation is performed through the CSCS system allocation tools. The default placement policy is first touch.

This behavior does not apply in the same way to memory allocated with `cudaMalloc`.

Course implications:

- Allocation mechanism is part of the performance model, not merely setup detail.
- First-touch placement should be made observable in at least one exercise or instructor demonstration.
- Unified addressing and cache coherence must not be presented as uniform bandwidth or latency. Placement and access locality still matter.
- Examples should explicitly state whether memory came from the system allocator, PyTorch/CUDA allocation, or another mechanism.
- Participants should distinguish logical accessibility, coherence, physical placement, migration behavior, and achieved bandwidth.
- A useful detour is to compare CPU/GPU access and placement for a system-allocated tensor or buffer against the conventional `cudaMalloc` path.

### Communication experiments

Before teaching rules such as “keep TP within the node,” establish an observed baseline:

- inspect the actual device topology;
- measure representative all-reduce, all-gather, reduce-scatter, all-to-all, and point-to-point operations;
- compare message-size regimes;
- distinguish startup/latency costs from bandwidth-dominated behavior;
- record the software stack and network integration used for the measurement.

The lesson should be that topology informs the initial hypothesis, while measurements determine the final strategy.

## Proposed narrative

Use one configurable decoder-only transformer that can switch between a dense FFN and an MoE FFN.

The day asks one progressively refined question:

> How should this model be mapped onto this machine, and what evidence shows that the mapping is appropriate?

Progression:

1. Start with a dense or locally routed baseline.
2. Identify why state sharding alone eventually stops being sufficient.
3. Shard individual transformer operations with tensor and sequence parallelism.
4. Partition layers and execution with pipeline parallelism.
5. Compose dimensions with `DeviceMesh`.
6. Replace dense FFNs with routed experts.
7. Place experts across devices and expose token-dispatch all-to-all communication.
8. Diagnose load imbalance, communication, and grouped expert computation.
9. Optimize one expert operation with `torch.compile` and Triton.
10. Compare the same operation with Mojo.
11. Select and defend an end-to-end configuration.

“3D parallelism” should be taught as composition of named mesh dimensions, not as an independent API or a single prescribed recipe.

## Day 3 learning objectives

Participants should be able to:

- estimate the memory occupied by parameters, gradients, optimizer state, and activations;
- distinguish data/state sharding from computation sharding;
- explain row-wise and column-wise tensor parallelism for attention and SwiGLU layers;
- read replicated, sharded, and partial DTensor layouts;
- associate TP/SP transitions with their required collectives;
- explain pipeline stages, microbatches, bubbles, and stage imbalance;
- construct and reason about an N-dimensional device mesh;
- explain top-k routing, active versus total parameters, expert capacity, token dropping, and dropless execution;
- explain expert-parallel dispatch, expert computation, and return communication;
- detect routing collapse and load imbalance from metrics;
- check numerical equivalence before accepting a performance result;
- decide whether a custom kernel is justified by measured behavior;
- distinguish an educational MoE implementation from a production-scale one.

## Detailed schedule

Assumption: 09:00-17:00, including a one-hour lunch and two breaks.

| Time | Block | Content and activity |
| --- | --- | --- |
| 09:00-09:20 | Opening challenge | Teams receive a model, memory constraint, shared resource envelope, and throughput target. They choose an initial strategy and predict the bottleneck before running anything. |
| 09:20-10:00 | MoE as a systems narrative | Dense transformer FFN; experts; top-k routing; active versus total parameters; sparsity; parameter memory; routing stability. Introduce the model used for the rest of the day. |
| 10:00-10:30 | Alps topology and memory model | Four GH200 modules, intra-node NVLink without NVSwitch, Grace/C2C and module links, Slingshot, system-allocated cache-coherent unified memory, first-touch placement, and the different `cudaMalloc` path. Connect topology to expected collective costs. |
| 10:30-10:45 | Break | |
| 10:45-11:20 | Tensor and sequence parallelism | Column/row sharding of attention and SwiGLU; DTensor; DeviceMesh; replicated/sharded/partial layouts; communication introduced by layout changes. |
| 11:20-12:15 | Lab A: TP diagnosis | Run TP degrees 1, 2, and 4 where resources permit. Verify numerical equivalence; record steady-state step time, peak HBM, and collective time; inspect a trace; explain why scaling does or does not improve. Optional sequence/loss parallel extension. |
| 12:15-12:30 | Evidence checkpoint | Each team reports one prediction confirmed and one disproved by the run. Consolidate a common cost model. |
| 12:30-13:30 | Lunch | |
| 13:30-14:05 | Pipeline parallelism and mesh composition | Manual versus traced splitting; GPipe and 1F1B; microbatches; bubbles; static shapes; stage balance; composition with TP and FSDP. |
| 14:05-14:45 | Lab B: pipeline trade-offs | Begin from an intentionally imbalanced split. Select split points and microbatch count. Compare throughput, peak activation memory, bubble fraction, and stage idle time. |
| 14:45-15:00 | Break | |
| 15:00-15:35 | MoE at scale and expert parallelism | Route, permute, all-to-all dispatch, grouped expert computation, reverse all-to-all, and unpermute. Cover capacity-based versus dropless execution, auxiliary losses, z-loss, and FP32 router computation. |
| 15:35-15:55 | Participant-selected detour | Vote among routing stability, context parallelism, DeepEP/MegaBlocks, unified-memory placement, distributed checkpoint reshaping, or deeper profiler analysis. |
| 15:55-16:35 | Kernel lab and Mojo comparison | Compare eager PyTorch, `torch.compile`, and a small Triton fused expert operation such as `silu(gate) * up`. Require correctness and numerical-error checks. Demonstrate the same kernel and benchmark shape from the Mojo uenv. |
| 16:35-16:55 | End-to-end decision challenge | Teams select a final DP/FSDP/TP/PP/EP configuration under memory, node-hour, and correctness constraints. Submit a compact evidence packet. |
| 16:55-17:00 | Close | State the selected mesh, limiting resource, rejected alternative, and next experiment. |

## Exercise design

### Common workload

The course workload should expose the following configuration dimensions:

- number of layers;
- hidden and intermediate dimensions;
- attention head count;
- vocabulary and sequence length;
- dense or MoE FFN;
- number of experts and top-k;
- capacity-based or dropless routing;
- DP/FSDP, TP, PP, and EP degrees;
- number of microbatches;
- eager, compiled, Triton, or comparison implementation;
- deterministic synthetic data or a locally cached token dataset.

The workload should support small correctness cases and larger performance cases without changing the model implementation.

### Evidence packet

Each exercise should produce:

1. Prediction before execution:
   - expected limiting resource;
   - expected dominant collective;
   - expected memory effect.
2. Exact configuration:
   - model dimensions;
   - parallel mesh;
   - batch and microbatch sizes;
   - dtype and compilation settings;
   - allocation path where memory placement matters.
3. Measurements:
   - warm-up/compile time reported separately;
   - steady-state step time and tokens/s;
   - peak HBM and relevant host memory;
   - collective or idle time;
   - output or loss parity.
4. Diagnosis:
   - bottleneck;
   - rejected alternative;
   - next experiment.

### Fault and perturbation cards

After the first successful run, assign each team one issue:

- router collapse onto a small number of experts;
- pipeline stage imbalance;
- increased TP degree reducing throughput;
- repeated recompilation caused by changing shapes;
- all-to-all dominating the MoE step;
- a fast but numerically incorrect custom kernel;
- poor first-touch placement;
- an incorrect assumption about the intra-node topology;
- restore from a checkpoint onto a different mesh.

The goal is diagnosis rather than code completion.

### Multi-node limitation

A four-GPU node cannot demonstrate DP, TP, and PP all with degree greater than one. A genuine 2 x 2 x 2 configuration needs eight GPUs.

Use:

- two active dimensions for most team exercises on one node;
- one shared, class-wide two-node run for full composition;
- recorded traces and metric bundles as a fallback and for comparison;
- short jobs with fixed bounds to suit the shared pool.

## Data, interpretation, evaluation, and instrumentation

The training data should serve two purposes:

1. drive a realistic language-model training loop and performance workload;
2. make model behavior inspectable through human-readable examples and known source labels.

A single unlabelled web-text stream is poor for the second purpose. Tiny synthetic data is easy to interpret but can make the systems exercise feel artificial. The recommended compromise is a small, balanced, source-labelled corpus prepared before the course.

### Recommended corpus: a labelled Dolma course mix

Prepare a fixed course subset from a pinned Dolma release. Dolma is an open, ODC-BY corpus containing web, academic, code, book, and encyclopedic material. Its document format preserves `id`, `text`, `source`, and source-specific metadata.

Select four human-distinguishable source families:

- narrative/books;
- encyclopedic text;
- question-and-answer or discussion text;
- source code and code-adjacent documentation.

The exact Dolma source names depend on the pinned release. Select and record them during data preparation rather than relying on mutable aliases.

Why this mixture:

- examples remain recognizable to participants;
- source labels provide a controlled variable for per-domain evaluation and routing analysis;
- narrative, encyclopedic, Q&A, and code sequences have visibly different token and structural patterns;
- the data remains close enough to real pretraining data for systems conclusions to be credible;
- one tokenizer, model, loader, and metric schema can serve dense and MoE runs.

Important interpretation caveat:

> A source label is a provenance/style label, not a semantic ground truth. An expert may specialize by punctuation, token frequency, position, syntax, or another feature instead of by source. The exercise should test specialization rather than assume it.

### Alternative and fallback: TinyStories

TinyStories is useful when the highest priority is fast emergence of coherent behavior in a very small model. Published experiments show coherent generation from models below 10 million parameters on its deliberately constrained story domain.

Advantages:

- small models learn visibly meaningful outputs;
- generations and attention patterns are easy to inspect;
- suitable for a fast pre-course smoke test or beginner track.

Limitations:

- synthetic and narrow;
- no natural multi-domain label for expert-specialization analysis;
- may teach a misleadingly easy language-modelling problem;
- dataset licensing must be handled separately from code/checkpoint licensing.

Recommendation: use the labelled Dolma mix as the main course corpus and keep a TinyStories configuration/checkpoint as a lightweight fallback and early development target.

### Dataset products

Prepare the data once; do not tokenize or download it during the course.

Create two compatible products:

- `course-mini`: small enough for local correctness checks and fast notebook exploration;
- `course-train`: large enough that repeated epochs and caching do not distort training/performance runs.

Both products should share:

- one pinned tokenizer and vocabulary;
- identical train/validation/probe splits;
- deterministic document ordering and shuffling seeds;
- document IDs and original source labels;
- provenance and licence metadata;
- fixed checksums and a data-version identifier.

Suggested batch schema:

```text
input_ids:  [batch, sequence]
target_ids: [batch, sequence]
source_ids: [batch, sequence] or [batch]
document_ids
attention/document-boundary metadata
```

The model and router must not receive `source_ids`; those labels exist only for sampling, evaluation, and post-hoc analysis.

For ordinary training, sequences may be packed for efficiency. Preserve a source ID per token when a packed sequence crosses document/source boundaries. For interpretability probes, use unambiguous single-document or single-source sequences.

### Model and checkpoint tiers

Use one implementation with separate configurations:

#### Semantic/interpretability configuration

- small enough for quick inference and activation capture;
- trained before the course until loss and generation are visibly meaningful;
- dense and MoE checkpoints at several training stages;
- deterministic probe prompts and held-out documents.

Participants continue training or fine-tune this model for a limited number of steps. They do not need to obtain language capability from random initialization during the workshop.

#### Systems/performance configuration

- same tokenizer, data schema, model code, and loss;
- dimensions scaled to expose HBM, compute, and communication limits;
- performance conclusions do not depend on reaching semantic convergence during the session.

This separation avoids two bad compromises: a toy model that cannot stress GH200s, or a large model that cannot become interpretable within course time.

### Interpretation activities

Use a fixed probe set so teams compare the same tokens and checkpoints.

Possible probes:

- narrative coreference and repeated entities;
- encyclopedia-style definitions and entity names;
- question/answer structure;
- code indentation, delimiters, keywords, and comments;
- repeated-token or copying patterns;
- deliberately mixed-domain prompts.

Collect only selected layers/heads/tokens. Capturing every activation from every rank would overwhelm storage and obscure the exercise.

#### Attention

Show:

- causal masks and attention matrices;
- head-to-head differences;
- changes between early and later checkpoints;
- local, delimiter, repeated-token, and long-range patterns.

Do not present attention weights as a complete explanation of model reasoning. They are one inspectable internal signal.

#### Hidden states

Useful low-dimensional summaries:

- per-layer activation norm and variance;
- cosine similarity between selected token states;
- PCA of selected token/document representations;
- residual-stream changes across layers;
- optional logit-lens projections as a detour.

Avoid UMAP/t-SNE as the primary evidence because visually compelling clusters can be sensitive to parameters. If used, retain PCA and quantitative distances beside them.

#### Expert routing and specialization

For every MoE layer, collect:

- tokens routed to each expert;
- mean router probability per expert;
- routing entropy;
- load coefficient of variation and max/mean load;
- capacity utilization and dropped-token count where applicable;
- source-by-expert routing matrix;
- representative high-probability token spans for each expert.

Useful questions:

- Does routing differ by source more than expected from the global token mix?
- Does specialization increase over training?
- Is apparent specialization caused by a few frequent tokens?
- Does stronger load balancing suppress measurable specialization?
- Does specialization survive a held-out split?
- Does an expert receive semantically related text or merely similar syntax/punctuation?

Mutual information or another association statistic between `source_id` and expert choice can be an optional summary, but it must be shown beside token counts and routing entropy. A single scalar should not substitute for inspecting the contingency matrix and examples.

### Evaluation suite

Keep evaluation small, deterministic, and local.

#### Always report

- held-out next-token loss;
- token-weighted perplexity;
- loss/perplexity per source;
- tokens evaluated;
- dense versus MoE active and total parameter counts;
- checkpoint/data/tokenizer identifiers.

#### MoE-specific evaluation

- auxiliary load-balancing loss;
- router z-loss;
- expert loads by layer;
- routing entropy;
- capacity and dropped-token statistics;
- source/expert association on the held-out probe set.

#### Qualitative evaluation

- generations from a fixed prompt set;
- side-by-side checkpoint comparisons;
- attention/routing views for fixed token sequences.

An LLM judge through the CSCS inference service can be an optional classroom comparison, not the primary score. Perplexity, routing statistics, and fixed samples are reproducible; a model judge introduces prompt/model/version variance.

Large benchmark suites such as `lm-evaluation-harness` are not a Day 3 requirement. They are unlikely to be informative for the small semantic model and would add setup/runtime unrelated to the scaling objectives.

## Observability from the first training step

Instrumentation should be part of the shared harness before parallel strategies are added. Retrofitting it after TP/PP/EP makes comparisons unreliable.

Use four layers of observability.

Expose explicit run modes so measurement and interpretation do not contaminate each other:

- `benchmark`: low-overhead scalar metrics; authoritative throughput result;
- `profile`: bounded PyTorch Profiler or Nsight capture; not used for headline throughput;
- `inspect`: selected activations, attention, and router decisions for the semantic model;
- `evaluate`: deterministic held-out and probe-set evaluation.

All modes use the same model/data configuration and run manifest. Expensive hooks must be absent, not merely ignored, in `benchmark` mode.

### Layer 0: always-on structured metrics

Low overhead; enabled for every run.

Write a machine-readable run manifest and JSON Lines step metrics. Console output is a human-readable view of the same data, not the authoritative record.

Run manifest:

- run ID, timestamp, course data version, tokenizer, checkpoint, and seed;
- source revision;
- PyTorch/CUDA/NCCL/container-or-uenv versions;
- model dimensions and total/active parameters;
- dtype, compile mode, activation checkpointing, and kernel implementation;
- nodes, ranks, mesh dimensions, device mapping, and Slurm job ID;
- relevant NCCL and network environment;
- warm-up and measurement windows.

Per-step or periodic metrics:

- loss and component losses;
- learning rate and gradient norm;
- tokens/s globally and per rank;
- step-time distribution, using the slowest rank as the distributed step time;
- data wait, host-to-device, forward, loss, backward, optimizer, and checkpoint time;
- allocated, reserved, and peak HBM;
- compile count/time and graph breaks where available;
- per-rank min/median/max for imbalance;
- MoE routing/load metrics;
- evaluation metrics when an evaluation window runs.

The current `notebooks/distributed/utils.py` is a useful seed but needs two corrections: host wall-clock timing must account for asynchronous CUDA execution, and distributed performance must not be summarized only by an average rank.

Use CUDA events for GPU intervals or synchronize only at deliberate measurement boundaries. Keep compilation/warm-up separate from steady-state throughput.

### Layer 1: bounded PyTorch Profiler traces

Use `torch.profiler` with a schedule such as:

```text
skip initial steps -> wait -> one warm-up step -> 2-3 active steps -> stop
```

Collect CPU and CUDA activities and add explicit ranges around:

- data loading and host-to-device movement;
- forward pass;
- attention;
- dense or MoE FFN;
- router/dispatch;
- loss;
- backward;
- optimizer;
- checkpointing.

Enable expensive options such as stack traces, shapes, and memory profiling only in dedicated profile runs. Do not use a profiler-enabled run as the authoritative throughput benchmark.

Trace layout:

```text
runs/<run-id>/
  run.json
  metrics.jsonl
  eval.json
  traces/rank-000.json.gz
  traces/rank-001.json.gz
  ...
```

Capture all ranks for short distributed trace windows. A rank-zero-only trace is useful for operator inspection but cannot explain cross-rank imbalance or collective synchronization.

PyTorch Profiler supports scheduled capture, trace callbacks, stack/shape/memory recording, execution traces, and memory timeline export. Pin the exact options to the course PyTorch release.

### Layer 2: distributed trace analysis and Nsight Systems

Use Holistic Trace Analysis or an equivalent pinned tool to summarize multi-rank PyTorch traces:

- compute, communication, memory, and idle fractions;
- kernel breakdown;
- per-rank outliers;
- communication/computation overlap;
- host wait and kernel launch gaps.

Use Nsight Systems for a small number of instructor-prepared or team-selected runs:

- validate CUDA/NCCL timelines;
- inspect stream concurrency;
- inspect CPU launch gaps;
- compare eager, compiled, Triton, and Mojo execution;
- validate whether communication overlaps compute.

Do not ask every team to capture full-day Nsight traces. They are large and expensive to interpret. Provide one guided trace and permit targeted captures after a scalar metric or PyTorch trace identifies a question.

### Layer 3: distributed failure diagnostics

Prepare a debug mode, disabled for performance measurements:

- NCCL/PyTorch distributed debug logging;
- ProcessGroupNCCL flight-recorder trace buffer;
- timeout-triggered dumps;
- desynchronization diagnostics;
- per-rank logs with stable names.

Use this mode for the controlled hang/desynchronization exercise. Debug logging and trace buffers can perturb performance and should not silently remain enabled in benchmark runs.

### Required phase labels

The same phase names must be used in scalar timers, PyTorch profiler ranges, and Nsight/NVTX ranges:

```text
data
h2d
forward
attention
ffn_dense
moe_router
moe_dispatch
moe_experts
moe_combine
loss
backward
optimizer
checkpoint
evaluation
```

Consistent naming makes it possible to move from a dashboard anomaly to a trace without translating between unrelated instrumentation schemes.

### Standard reports

Every benchmark matrix should generate:

1. throughput and step time versus GPU count/parallel strategy;
2. peak HBM per rank;
3. compute/communication/idle breakdown;
4. communication/computation overlap;
5. per-rank or per-stage imbalance;
6. training and held-out loss versus tokens;
7. per-source perplexity;
8. expert-load and source-by-expert heatmaps for MoE;
9. correctness difference from the single-rank/eager reference;
10. compile and warm-up overhead reported separately.

The end-to-end challenge should consume these reports rather than asking participants to assemble ad hoc evidence from log text.

## Triton and Mojo

### Triton core exercise

Use a small operation attached to the MoE expert MLP, not an unrelated tutorial kernel. A fused gated activation is a suitable first candidate:

```text
output = silu(gate) * up
```

Compare:

- eager PyTorch;
- `torch.compile`;
- Triton;
- Mojo demonstration.

All implementations must use identical shapes, dtype, warm-up, timing, and correctness tolerances.

A valid conclusion is that the compiler already generates an adequate kernel. Custom kernels require a measured benefit that justifies additional maintenance and portability costs.

### Mojo role

The working Mojo uenv makes the demo a committed block rather than a contingency.

The core course should still use Triton because it is directly integrated into the PyTorch compilation ecosystem and allows participants to stay in Python. Mojo provides a useful comparison of language model, layout control, compilation, and explicit GPU programming.

Potential optional task for fast teams:

- run the prewritten Mojo kernel;
- vary one shape or tile parameter;
- compare correctness and performance with the Triton implementation;
- explain the abstraction and tooling differences rather than declaring a universal winner.

## Agent-enabled pedagogy

Agents are explicitly permitted.

Recommended loop:

```text
predict -> ask agent -> run -> measure -> challenge explanation -> defend
```

Agents may:

- inspect documentation and repository code;
- propose sharding or pipeline plans;
- implement mechanical configuration changes;
- help interpret traces;
- generate plots;
- review a diagnosis.

Participants remain responsible for:

- approving resource use;
- predicting behavior before execution;
- validating correctness;
- separating evidence from a plausible narrative;
- explaining and defending the final decision.

A useful comparison table:

| Question | Human prediction | Agent prediction | Observation | Final explanation |
| --- | --- | --- | --- | --- |

### Agent safety on Alps

Provide a course job wrapper with:

- explicit account and reservation;
- fixed node and wall-time maxima;
- dry-run by default;
- one outstanding job per team;
- explicit human approval before submission;
- bounded log and artifact paths;
- no unrestricted job-submission loops.

Inference-service keys should have course-appropriate token budgets and should not be embedded in repositories or shell startup files.

## Runtime and environment plan

Canonical paths:

- the pinned Alps extended PyTorch container;
- the existing Mojo uenv for the comparison block;
- Clariden JupyterHub;
- controlled Slurm jobs;
- the CSCS inference service for agentic work.

Canonical PyTorch image:

```text
jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/pytorch-cuda:26.07-py3-alps7-dev-14ac42497583f60a1dd156230f00af8f
```

Policy:

- Resolve and record the image's immutable digest in the new course environment.
- Use the image-provided PyTorch, CUDA, NCCL, and Alps network stack; do not shadow them with `uv` packages.
- Use `uv` for pinned course-owned Python dependencies, Jupyter, analysis, and authoring tools.
- Use the Mojo uenv for the comparison block.
- Use JupyterLab for instructions, predictions, plots, and short exploration.
- Keep distributed execution in version-controlled Python scripts launched through a controlled Slurm or `torch.distributed.run` path.
- Pre-pull the image and pre-stage all data.
- Keep compiler caches off shared home storage unless a validated course-specific cache design is provided.
- Maintain one participant-facing PyTorch execution path. A recovery environment may be tested internally but must not create duplicate instructions.

## Relationship to Days 1 and 2

### Course-level risk

Last year's repository describes a beginner-friendly introduction, while the proposed three-day outline is an advanced PyTorch performance and distributed-systems course. A mixed cohort needs a preparation path.

Reuse existing introductory notebooks as prework or an optional preparation session:

- tensors, modules, and training loops;
- transformer block structure;
- attention and FFNs;
- basic Slurm and Jupyter operation.

### Day 1

Use the same transformer workload and establish:

- eager correctness and performance baseline;
- profiler workflow;
- mixed precision;
- activation memory and checkpointing;
- `torch.compile`;
- input-pipeline isolation;
- standard measurement protocol.

Every performance claim should carry output/loss parity, dtype policy, warm-up protocol, peak memory, throughput, and profiler evidence.

### Day 2

Modernize the existing distributed lab:

- use the common transformer instead of a small CIFAR CNN for the main scaling experiment;
- teach DDP, then FSDP2;
- introduce DeviceMesh before Day 3;
- connect collectives to observed DDP/FSDP behavior;
- use Distributed Checkpoint for save, restore, and resharding;
- include one controlled distributed failure or hang diagnosis.

The existing `notebooks/distributed` material provides a useful structural base, but its model is too small to make FSDP benefits compelling. The slides and helper code also use the older FSDP1 wrapper and should be migrated to FSDP2.

### Three-day conceptual arc

| Day | Central question |
| --- | --- |
| Day 1 | How efficiently can one accelerator execute the model? |
| Day 2 | How do replicas cooperate, and how do we shard training state? |
| Day 3 | How do we partition computation, and what changes when computation is conditional? |

## Reuse plan for existing slides and notebooks

The course does not need to be rebuilt from scratch. The existing repository already contains most of the conceptual bridge from DDP/FSDP to TP/PP/3D parallelism, a usable transformer-component sequence, and a distributed execution/benchmark harness.

The main work is to reorganize and modernize that material around one transformer/MoE workload. The genuinely new parts are DTensor-based implementations, MoE/expert parallelism, the kernel comparison, the Alps unified-memory material, and evidence-oriented exercises.

### Slide material

| Existing material | Reuse level | Proposed use | Required adjustment |
| --- | --- | --- | --- |
| `slides/src/slidev-theme-cscs/`, `slides/src/components/`, and the root Slidev assembly | High | Keep the existing presentation toolchain, CSCS branding, layouts, code highlighting, and build/deployment path. | Create a new section for large-scale models and revise the course title, outline, dates, and logistics. |
| `slides/src/2.3-distributed/section-slides.md`: DDP animation and data-parallel overview | High for Day 2 | Retain as the visual explanation of replication and gradient synchronization. | Shorten after the first complete animation; connect its all-reduce directly to measured traces. |
| The Adam, ZeRO, and FSDP slides in `2.3-distributed` | Medium to high for Day 2 | Retain the memory-accounting and state-sharding motivation that leads into Day 3. | Update terminology and code from FSDP1 to FSDP2; make the memory model explicit; remove statements tied to the old wrapper API. |
| Tensor/operator parallelism slides in `2.3-distributed` | High for Day 3 | Use as the conceptual opening for tensor parallelism. The operator-parallel image is already suitable for the “split computation, not only state” message. | Add concrete row-wise and column-wise transformer examples, DTensor layouts, and collectives. Replace the claim that specialized third-party frameworks are required with current PyTorch-native capabilities and API maturity caveats. |
| Pipeline-parallelism slides and `pipeline_parallelism.png`/`pipeline_ds_*.png` | High for Day 3 | Reuse the stage and microbatch visual model. | Add GPipe versus 1F1B timelines, bubble fraction, stage balance, and current `torch.distributed.pipelining` terminology. |
| 3D-parallelism slide and `3d_parallelism.png` | Medium to high for Day 3 | Reuse as the transition from separate techniques to composition. | Redraw or annotate it with named `DeviceMesh` dimensions and the world-size product. Include EP as an additional dimension rather than implying that “3D” is a fixed final architecture. |
| NCCL and collective material in `2.3-distributed` | High across Days 2 and 3 | Keep the collective vocabulary and use it when reading profiler traces. | Add all-to-all for MoE, map each collective to DDP/FSDP/TP/PP/EP, and replace API-only description with measured latency/bandwidth regimes. |
| Slingshot, dragonfly, libfabric, and aws-ofi-nccl slides in `2.3-distributed` | High, with hardware corrections | Reuse the Alps-specific network stack and existing `nw-local.png`, `nw-topology.png`, and `ofi.png` assets. | Add intra-node H100 NVLink, absence of NVSwitch, Grace/C2C and module links, and the system-allocated unified-memory/first-touch model. Check all feed speeds and topology labels against the current system. |
| `notebooks/transformers/slides-nlp_with_transformers.pdf` | Medium | Reuse selected transformer recap material: multi-head attention, position-wise FFN, masked attention, and decoder-only architecture. It is useful as prework or a short Day 3 recap. | The editable source is not present in this repository. Locate it if possible. Otherwise port only the needed 4-6 slides into Slidev rather than rebuilding or presenting the full NLP deck. Modernize the FFN to SwiGLU where it connects to TP and MoE. |
| `slides/src/1.2-pytorch-overview/section-slides.md` | High for prerequisites/Day 1 | Reuse tensors, modules, parameters, autograd, training-loop, and state-dict material. | Remove duplication from the three-day live schedule by assigning basic portions as prework for experienced attendees. |
| `slides/src/3.1-more_on_training/section-slides.md` | Medium for Day 1/prework | Reuse `state_dict`, train/eval, and residual-connection material. | Distributed checkpointing needs new DCP material; ordinary `torch.save` slides are only the conceptual starting point. |
| `slides/src/1.1-introduction`, `2.1-featurisation`, and `2.2-cnn` | Low for Day 3 | Keep as optional beginner preparation and examples of the existing teaching style. | Do not place them in the main Day 3 path. |

Existing public image assets with direct reuse value:

- `ddp1.png` through `ddp9.png`;
- `allreduce_ring_1.png` through `allreduce_ring_9.png`;
- `ZeRO.png` and `fsdp.png`;
- `operator_parallelism.png`;
- `pipeline_parallelism.png` and `pipeline_ds_*.png`;
- `data_parallelism.png` and `3d_parallelism.png`;
- `nw-local.png`, `nw-topology.png`, and `ofi.png`.

The existing images cover most of the first visual explanation. New diagrams are needed only for DTensor layouts, current Alps intra-node topology/memory placement, pipeline schedules, MoE dispatch, and the final multi-dimensional mesh.

### Notebook and code material

| Existing material | Reuse level | Proposed use | Required adjustment |
| --- | --- | --- | --- |
| `notebooks/transformers/3_self-attention.ipynb` | Medium | Use as a prerequisite or short attention-shape recap before explaining Q/K/V and output-projection sharding. The matrix visualizations and `utils.plot_matrix` remain useful. | Remove the online BERT dependency from the performance path; use a deterministic local transformer block. The notebook currently creates new linear layers inside the attention helper and is educational code, not a reusable training module. Add causal attention and annotate global versus per-TP-rank shapes. |
| `notebooks/transformers/4_feedforward-layer.ipynb` | High as a starting point | This is the strongest seed for the MoE narrative. Start with its position-wise FFN, change it to a modern gated/SwiGLU FFN, then duplicate it into experts and add routing. | Replace the BERT/GELU-specific setup, remove downloads, turn the class into a tested model component, and add shape/memory annotations. |
| `notebooks/transformers/1_tokenizers.ipynb`, `2_embeddings.ipynb`, and positional-encoding notebooks | High for prework; low for the Day 3 core | Reuse to make the transformer workload understandable to less experienced participants. | Keep off the critical performance path and cache any required model/tokenizer artifacts. |
| `notebooks/distributed/distributed_utils.py` | High | Retain the environment-based process-group setup and cleanup pattern for all distributed exercises. | Complete the scaffold, improve failure handling, and make device/backend initialization compatible with the pinned PyTorch version. Add named mesh creation above this layer rather than replacing it. |
| `notebooks/distributed/run.sh` | High | Retain the Slurm rank mapping, bounded job, `srun` launch, and environment-variable pattern. | Parameterize account/reservation/environment and exercise configuration. Preserve the bounded course wrapper rather than allowing agents to submit arbitrary scripts. |
| `notebooks/distributed/edf.toml` | High if a container is canonical | Retain the image/mount/workdir pattern and the aws-ofi-nccl annotations. | Update the pinned image and paths. If a PyTorch uenv becomes canonical, keep this as the tested recovery path rather than maintaining duplicate participant instructions. |
| `notebooks/distributed/main.py`, `train.py`, and `.solution/` variants | Medium to high structurally | Reuse the entrypoint, student/solution split, optimizer/training-loop organization, and rank-zero reporting. | Replace the CNN and classification-specific validation with the shared transformer/MoE model and deterministic token objective. Move strategy selection into configuration for DDP/FSDP/TP/PP/EP. |
| `notebooks/distributed/ddp_utils.py` | High for Day 2 | Retain the small wrapper exercise and replica mental model. | Instantiate the shared transformer and update device setup. |
| `notebooks/distributed/fsdp_utils.py` | Low as implementation; high as teaching contrast | Use the existing FSDP1 code to explain what changed and why the course now uses FSDP2. | Rewrite around `fully_shard`, per-layer sharding, DTensor parameters, and current mixed-precision policies. Do not extend the old `FullyShardedDataParallel` wrapper. |
| `notebooks/distributed/data.py` | Medium for Day 2 | Reuse the `DistributedSampler`, rank-zero preparation, barrier, and global/local batch-size lessons. | Do not download CIFAR during the scaling labs. Replace with deterministic synthetic tokens or a pre-staged local token dataset. |
| `notebooks/distributed/utils.py` | Medium to high | Reuse rank-aware logging, reduced metrics, and CUDA memory reporting. | Add CUDA synchronization/events where timing requires it; report tokens/s, compile/warm-up time, MFU or achieved FLOPs where meaningful, and per-rank imbalance. |
| `notebooks/distributed/run_benchmarks.py` | Medium as workflow | Reuse the configuration-matrix concept and generated bounded Slurm scripts. | Replace automatic submission of eight jobs with a course coordinator or guarded queue. Drive experiments from a manifest and record full configuration metadata. |
| `notebooks/distributed/collect_results.py` and `plot_results.py` | Medium | Reuse the log-to-table-to-plot workflow. | Move from filename/regex assumptions to structured JSON/CSV metrics. Correct the meaning of per-rank peak memory, add communication and idle-time plots, and retain correctness fields. |
| `notebooks/pipelines/` | None for pipeline parallelism | Keep for model-inference demonstrations elsewhere in the course. | The directory contains Hugging Face task pipelines, not pipeline parallel training. Rename or clearly distinguish it to avoid participant confusion. |
| `notebooks/bert_squad/`, `cnn/`, and `more_on_training/` | Low for Day 3 | Retain as beginner/fine-tuning material and optional preparation. | Do not adapt these into the large-scale systems exercises. |

### Build the new workload by composition

The lowest-effort path is not to create independent TP, PP, MoE, and kernel notebooks. Build one small package or script set from existing components:

1. Start with the FFN and attention concepts in `notebooks/transformers`.
2. Put them into a deterministic decoder-only block with no required external download.
3. Place that model behind the launch, logging, solution, and benchmark structure from `notebooks/distributed`.
4. Add one strategy/configuration layer for DDP/FSDP2/TP/PP/EP.
5. Let notebooks explain, launch, and analyze runs; keep the distributed implementation in importable Python files.
6. Reuse the same correctness cases, metric schema, and plotting path for every strategy.

This avoids five unrelated notebooks that each reimplement initialization, timing, logging, and Slurm integration.

### New material that is still required

No current repository material implements the following:

- `DeviceMesh`, DTensor placements, and native tensor/sequence/loss parallelism;
- FSDP2;
- `torch.distributed.pipelining` and current schedules;
- a decoder-only shared performance model;
- MoE routing, load metrics, auxiliary losses, and expert parallelism;
- all-to-all dispatch and grouped expert computation;
- Triton kernels;
- Mojo comparison material;
- system-allocator versus `cudaMalloc` unified-memory experiments;
- evidence packets, fault cards, and agent guardrails.

These do not all need original implementations:

- adapt the pinned PyTorch TP, DeviceMesh, FSDP2, and pipelining tutorials;
- adapt the router and stability progression from nanoMoE;
- use TorchTitan as the reference for composition and production configuration rather than copying its training stack;
- adapt an official Triton tutorial kernel structure to the selected expert operation;
- reuse the working Mojo uenv and an existing validated Mojo kernel if available.

Keep an attribution/version note beside adapted upstream code. Pin the source revision used by the course so that experimental PyTorch APIs do not drift underneath the exercises.

### Planning estimate

Approximate reuse for Day 3:

- slide framework, visual assets, and conceptual slides: 50-60%;
- notebook/job/measurement infrastructure: 40-50%;
- Day 3 core distributed-model code: mostly adapted from official examples and nanoMoE, with a smaller CSCS-specific integration layer;
- genuinely CSCS-specific new content: current GH200 topology and unified-memory behavior, Slurm/environment integration, measured traces, and the end-to-end challenge.

The highest-value refactor is therefore a shared transformer/MoE harness, not a new slide deck or a collection of standalone notebooks.

## Source and framework notes

### Existing repository

Relevant existing material:

- `slides/src/2.3-distributed/section-slides.md`: DDP, FSDP, TP, PP, 3D parallelism, NCCL, and Alps network background.
- `notebooks/distributed/`: DDP/FSDP lab and scaling harness.
- `notebooks/transformers/`: transformer component material suitable for prework or recap.
- `notebooks/more_on_training/`: previous optimization exercises.

### MoE references

The nanoMoE article is a strong source for the algorithmic progression:

- expert layers;
- softmax top-k routing;
- capacity and token dropping;
- load-balancing loss;
- router z-loss;
- full-precision router computation;
- training-stability experiments.

It should be supplemented with production execution concepts:

- token permutation and bucketing;
- all-to-all dispatch and return;
- grouped or block-sparse expert computation;
- expert placement;
- expert-parallel checkpointing;
- composition with other parallel dimensions.

Useful references:

- nanoMoE: <https://cameronrwolfe.substack.com/p/nano-moe>
- PyTorch MoE scaling article: <https://pytorch.org/blog/training-moes/>
- PyTorch tensor parallel API: <https://docs.pytorch.org/docs/main/distributed.tensor.parallel.html>
- PyTorch TP tutorial: <https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html>
- PyTorch pipeline parallelism: <https://docs.pytorch.org/docs/main/distributed.pipelining.html>
- PyTorch DeviceMesh: <https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html>
- PyTorch FSDP2 tutorial: <https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html>
- TorchTitan: <https://github.com/pytorch/torchtitan>
- Dolma dataset and toolkit: <https://allenai.github.io/dolma/>
- Dolma document/source schema: <https://github.com/allenai/dolma/blob/main/docs/data-format.md>
- TinyStories paper: <https://arxiv.org/abs/2305.07759>
- PyTorch Profiler recipe: <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>
- PyTorch Profiler API: <https://docs.pytorch.org/docs/stable/profiler.html>
- Holistic Trace Analysis tutorial: <https://docs.pytorch.org/tutorials/beginner/hta_intro_tutorial.html>
- ProcessGroupNCCL diagnostic environment variables: <https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html>
- CSCS Clariden documentation: <https://docs.cscs.ch/clusters/clariden/>
- CSCS PyTorch documentation: <https://docs.cscs.ch/software/ml/pytorch/>
- CSCS Jupyter documentation: <https://docs.cscs.ch/access/jupyterlab/>
- CSCS inference service: <https://docs.cscs.ch/services/inference/api/>
- CSCS coding-agent guidance: <https://docs.cscs.ch/guides/coding-agents/>
- Mojo requirements and GPU support: <https://mojolang.org/docs/requirements/>

## Open questions

- Course date and exact PyTorch release to pin.
- Number of participants and expected team size.
- Number of simultaneously available Clariden nodes.
- Whether a two-node class-wide allocation can be guaranteed for the composition exercise.
- Exact immutable digest and supported Jupyter/Slurm launch configuration for the selected Alps extended PyTorch image.
- Exact system-allocation API and PyTorch integration to use for the unified-memory demonstration.
- Which intra-node communication paths NCCL selects for representative operations and message sizes.
- Whether the Mojo comparison remains instructor-only or becomes an optional participant exercise.
- Whether to use a minimal native-PyTorch course harness, a pinned TorchTitan subset, or both.
- Which participant-selected detours instructors are prepared to support.
- How inference-service keys and budgets will be issued for course accounts.
- Exact pinned Dolma release, source families, token counts, licence/provenance bundle, and tokenizer.
- Target sizes and pretraining budgets for the semantic dense and MoE checkpoints.
- Which profiler/trace analysis versions and output viewers are supportable in the canonical environment.

## Preparation and pilot checklist

- Pin the PyTorch environment and record all relevant versions.
- Validate TP, PP, FSDP2, compilation, and Triton in that environment.
- Validate the Mojo uenv and comparison kernel.
- Measure intra-node and inter-node collectives.
- Verify the actual NCCL/libfabric/Slingshot path.
- Pilot system-allocated unified memory and first-touch placement examples.
- Calibrate model sizes so correctness runs are short and performance runs expose the intended bottlenecks.
- Test one-node and two-node job paths.
- Pre-stage datasets, tokenizers, images, and environment artifacts.
- Build and checksum `course-mini` and `course-train` tokenized datasets with preserved source/document IDs.
- Train dense and MoE semantic checkpoints and freeze a deterministic interpretability probe set.
- Define the run manifest, JSON Lines metric schema, trace directory convention, and standard report notebook.
- Pilot profiler overhead and storage volume for one-rank and all-rank trace modes.
- Create recorded trace and metrics fallbacks for every resource-dependent exercise.
- Validate the course job-submission guardrails.
- Run the complete schedule with realistic queue and compilation overhead.

## Change log

- 2026-09-16: Initial living document. Added Day 3 schedule, MoE narrative, agent-enabled exercise model, environment plan, and recommendations for Days 1 and 2.
- 2026-09-16: Corrected the GH200 node model: intra-node H100 NVLink is present, NVSwitch is absent, Grace/C2C and module interconnects matter, and system-allocated memory can expose cache-coherent unified node memory with first-touch placement; this differs from `cudaMalloc` allocation.
- 2026-09-16: Added a slide/notebook reuse audit, identified the shared transformer/MoE harness as the main refactor, and separated reusable infrastructure from genuinely new Day 3 material.
- 2026-09-16: Added the labelled Dolma course-corpus proposal, TinyStories fallback, semantic versus systems configurations, interpretation/evaluation probes, and a four-layer observability design.
- 2026-09-16: Made the Alps extended PyTorch container the canonical runtime and linked Day 3 to the clean-slate, full three-day rebuild and authorship-preserving migration plan.
- 2026-09-16: Replaced the proposed course-wide attribution manifest with a Git-native migration convention: unchanged move/copy commits first, adaptations in later commits, with separate notices retained only for external sources and licences.
