# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay and episode data structures for RL Token training."""

from groot_rlt.collate import collate_rlt_batch
from groot_rlt.episode_schema import (
    ChunkBehaviorSource,
    ChunkDecisionRecord,
    EpisodeSource,
    EpisodeStepRecord,
    HandoffSource,
    InterventionRecord,
    RLTEpisodeRecord,
    StepBehaviorSource,
    StepReferenceSource,
    TerminalLabelSource,
    validate_episode_components,
)
from groot_rlt.episode_transition_builder import (
    EpisodeTransitionBuilder,
    build_rlt_transitions_from_episode,
)
from groot_rlt.networks import (
    RLTActor,
    RLTCritic,
    RLTDoubleCritic,
    RLTNetworkConfig,
    fixed_std_log_prob_normalizer,
    gaussian_entropy_per_sample,
    make_rlt_networks,
    reference_regularization_loss,
    td3_min_target,
)
from groot_rlt.replay_buffer import RLTReplayBatch, RLTReplayBuffer, make_replay_buffer
from groot_rlt.replay_schema import (
    BehaviorSource,
    CollectionStage,
    CriticalPhaseSegment,
    ReferenceSource,
    RLTTransition,
    TerminalOutcome,
    compose_train_reference_chunk,
)
from groot_rlt.train import (
    RLTTrainConfig,
    discounted_reward_sum_from_chunk,
    soft_update_target_network,
    update_actor,
    update_critic,
)
from groot_rlt.trainer import RLTTrainer, RLTTrainerConfig, RLTTrainerRollout, RLTTrainerState
from groot_rlt.warmup import (
    FileTensorResolver,
    RLTWarmupConfig,
    RLTWarmupSummary,
    WarmupEpisodeComponents,
    build_warmup_replay,
    build_warmup_transitions,
    load_warmup_episode_directory,
    load_warmup_replay,
    save_warmup_replay,
    write_warmup_summary_json,
)

__all__ = [
    "BehaviorSource",
    "ChunkBehaviorSource",
    "ChunkDecisionRecord",
    "CollectionStage",
    "CriticalPhaseSegment",
    "EpisodeSource",
    "EpisodeStepRecord",
    "EpisodeTransitionBuilder",
    "FileTensorResolver",
    "HandoffSource",
    "InterventionRecord",
    "ReferenceSource",
    "RLTActor",
    "RLTCritic",
    "RLTDoubleCritic",
    "RLTNetworkConfig",
    "RLTReplayBatch",
    "RLTReplayBuffer",
    "RLTEpisodeRecord",
    "RLTTrainConfig",
    "RLTTrainer",
    "RLTTrainerConfig",
    "RLTTrainerRollout",
    "RLTTrainerState",
    "RLTTransition",
    "RLTWarmupConfig",
    "RLTWarmupSummary",
    "StepBehaviorSource",
    "StepReferenceSource",
    "TerminalOutcome",
    "TerminalLabelSource",
    "WarmupEpisodeComponents",
    "build_rlt_transitions_from_episode",
    "build_warmup_replay",
    "build_warmup_transitions",
    "collate_rlt_batch",
    "compose_train_reference_chunk",
    "discounted_reward_sum_from_chunk",
    "fixed_std_log_prob_normalizer",
    "gaussian_entropy_per_sample",
    "load_warmup_episode_directory",
    "load_warmup_replay",
    "make_rlt_networks",
    "make_replay_buffer",
    "reference_regularization_loss",
    "save_warmup_replay",
    "soft_update_target_network",
    "td3_min_target",
    "update_actor",
    "update_critic",
    "validate_episode_components",
    "write_warmup_summary_json",
]
