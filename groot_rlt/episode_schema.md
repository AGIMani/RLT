# RLT Episode Directory Layout

This document describes the recommended on-disk layout for one collected RLT
episode. It sits next to `episode_schema.py` because the files below map directly
to the episode-level dataclasses defined there.

```text
episode_000000/
  episode.json                 # RLTEpisodeRecord: episode-level metadata
  steps.parquet                # EpisodeStepRecord: one row per frame
  chunk_decisions.parquet      # ChunkDecisionRecord: one row per decision_t
  interventions.jsonl          # InterventionRecord: one row per intervention segment
  videos/
    ego_view.mp4
    wrist_view.mp4
  tensors/
    rl_tokens/
    proprio/
    executed_actions/
    vla_reference_chunks/
    actor_proposed_chunks/
    human_correction_chunks/
```

## File Responsibilities

### `episode.json`

Stores one `RLTEpisodeRecord`.

It contains episode-level metadata that should not be repeated on every frame:

- identity: `episode_id`, `episode_index`, `task_id`, `instruction`;
- source: `demo`, `vla_warmup`, `online_rl`, `human_intervention`, or `mixed`;
- data shape: `length`, `fps`, `state_dim`, `action_dim`;
- robot and environment: `robot_type`, `env_id`;
- data locations: `data_path`, `video_paths`;
- chunk configuration: `chunk_len`, `vla_horizon`, `chunk_stride`;
- critical phase timing: `episode_start_t`, `base_vla_prefix_end_t`, `handoff_t`,
  `terminal_t`;
- sparse terminal label: `terminal_outcome`, `terminal_label_source`;
- optional provenance: VLA checkpoint, RL token encoder checkpoint, actor
  checkpoint, policy version, and raw episode id/path.

Required invariant:

```text
0 <= base_vla_prefix_end_t <= handoff_t < terminal_t < length
```

### `steps.parquet`

Stores many `EpisodeStepRecord` rows, one row per frame `t`.

Each row keeps the per-frame time series and stable tensor keys:

- time index: `t`, `frame_index`, `timestamp_s`;
- observation references: `obs_key`, `image_keys`, `proprio_key`, `rl_token_key`;
- executed action reference: `executed_action_key`;
- control provenance: `behavior_source`;
- reference provenance: `reference_source`;
- intervention marker: `intervention`;
- chunk alignment: `is_decision_step`, `decision_t`, `chunk_offset`;
- sparse reward fields: `reward`, `done`;
- optional per-step references:
  `vla_reference_action_key`, `actor_proposed_action_key`,
  `human_correction_action_key`, `intervention_id`.

Required invariants:

```text
len(steps) == episode.length
steps are sorted and contiguous by t: 0, 1, ..., length - 1
chunk_offset == t - decision_t
done is true only at terminal_t
non-terminal reward == 0
terminal reward == terminal_outcome.reward
```

For intervention rows:

```text
behavior_source == human_intervention
reference_source == human_intervention
human_correction_action_key is required
intervention_id is required
```

### `chunk_decisions.parquet`

Stores many `ChunkDecisionRecord` rows, one row per policy or VLA decision
boundary.

Each row records the chunk-level proposals needed to derive replay transitions:

- decision boundary: `decision_t`;
- state references at the decision boundary: `obs_key`, `rl_token_key`,
  `proprio_key`;
- VLA reference chunk: `vla_reference_chunk_key`;
- chunk configuration: `vla_horizon`, `chunk_len`, `executed_prefix_len`;
- chunk behavior source: `vla`, `actor`, `human`, or `mixed`;
- optional proposal chunks: `actor_proposed_chunk_key`,
  `human_correction_chunk_key`.

Required invariants:

```text
handoff_t <= decision_t < terminal_t
chunk_len == episode.chunk_len
vla_horizon == episode.vla_horizon
each decision_t appears at most once
0 < executed_prefix_len <= chunk_len
```

If the chunk was produced by the actor, `actor_proposed_chunk_key` is required.
If it was produced by a human correction, `human_correction_chunk_key` is
required. Mixed chunks must keep at least one actor or human provenance key.

### `interventions.jsonl`

Stores many `InterventionRecord` rows, one row per contiguous human takeover or
correction segment.

Each row records:

- identity: `episode_id`, `intervention_id`;
- interval: `start_t`, `end_t_exclusive`;
- optional takeover frame: `takeover_t`;
- human correction chunk: `human_action_chunk_key`;
- control state: `control_source`, `authority_state`;
- optional metadata: `operator_id`, `reason`.

The interval is half-open:

```text
start_t <= t < end_t_exclusive
```

Required invariants:

```text
0 <= start_t < end_t_exclusive <= episode.length
start_t < terminal_t
takeover_t, when present, lies inside the interval
each intervention_id appears at most once
```

## Tensor And Video Storage

Large arrays and media should be stored by path/key instead of embedded directly
inside the metadata records.

Recommended tensor folders:

- `tensors/rl_tokens/`: cached RL token features, keyed by step;
- `tensors/proprio/`: proprioception arrays, keyed by step;
- `tensors/executed_actions/`: executed low-level actions, keyed by step or chunk;
- `tensors/vla_reference_chunks/`: VLA reference action chunks;
- `tensors/actor_proposed_chunks/`: actor proposal chunks during RL rollout;
- `tensors/human_correction_chunks/`: human correction chunks during intervention.

Recommended video folders:

- `videos/ego_view.mp4`;
- `videos/wrist_view.mp4`.

`episode.json`, `steps.parquet`, and `chunk_decisions.parquet` should reference
these assets through stable keys such as `rl_token_key`,
`vla_reference_chunk_key`, or `human_correction_action_key`.

## Relationship To Replay

This episode layout is the collection-time source of truth. Replay transitions
are derived later from these records.

The replay builder should use:

- `RLTEpisodeRecord` for critical phase and terminal sparse label metadata;
- `EpisodeStepRecord` for `x`, `x'`, executed actions, rewards, and done flags;
- `ChunkDecisionRecord` for reference chunks, actor proposals, and chunk
  alignment;
- `InterventionRecord` for replacing the training reference with human
  correction chunks inside intervention segments.

