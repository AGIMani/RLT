# GR00T RLT / Teleop integration contract

This note records the read-only interface review of `zhangbt@node0:~/Teleop`
(the actual directory name is `~/Teleop`) and defines the boundary for a future
live Groot-RLT adapter. It is deliberately not a real-robot launch guide.

## Decision

Do not connect Machine A's 26D VLA reference directly to the Teleop hardware
path. Machine A keeps that complete reference for the frozen 400k checkpoint,
while Machine B explicitly projects it to the real 19D EEF-and-hand command
before actor/critic, replay, normalization, fallback, or execution. Keep the
existing Pika/Agilex ROS adapter unchanged and add a separate Teleop policy
adapter for this contract.

Teleop already owns the safety-critical parts of the loop:

- operator input and human/policy authority;
- safety hold and guarded rollout;
- RTC/history stitching;
- command timing and robot output;
- action, policy, authority, and episode capture.

Groot-RLT should therefore integrate as a policy provider. It must not bypass
Teleop's authority mux or call the CAN/robot drivers directly.

## Interfaces verified on node0

The following paths and contracts were inspected without running the robot:

| Teleop path | Contract |
| --- | --- |
| `src/teleop_stack/policies/base.py` | `PolicyObservation`, `PolicyActionChunk`, `PolicyInterface` |
| `src/teleop_stack/session/action_authority.py` | `ActionAuthorityMux`, `AuthorityDecision`, intervention IDs and control-source provenance |
| `src/teleop_stack/models.py` | `SingleArmTeleopCommand` and embodiment-neutral `CommandEnvelope` |
| `src/teleop_stack/policies/groot_policy.py` | native GR00T observation construction and action-to-command conversion |
| `src/teleop_stack/data_capture/exporters/rlt_episode.py` | offline export shaped like this repository's RLT episode sidecars |

The intended live shape is:

```text
PolicyObservation
  -> GrootRltTeleopPolicy.infer(...)
       -> Machine A: z_rl + 26D physical GR00T reference + 26D proprio
       -> Machine B: validate source metadata and project 26D -> 19D
       -> Machine B: 19D actor/critic/replay refinement
       -> denormalize exactly once
       -> validated 19D EEF-and-hand command converter
  -> PolicyActionChunk
  -> PolicyTrajectoryManager
  -> ActionAuthorityMux
  -> rollout guard / robot adapter
```

`ActionAuthorityMux` remains outside the RLT policy. Human takeover must cancel
or supersede policy authority there, while preserving the policy proposal and
reference action for replay provenance.

## Action-space contract

The reference, executable action, and proprio contracts are intentionally
different:

```text
Machine A reference: eef_9d[9] + hand_joint_target[10] + arm_joint_target[7] = 26D
Machine B action:    eef_9d[9] + hand_joint_target[10]                         = 19D
Machine B proprio:  eef_9d[9] + hand_joint_pos[10] + arm_joint_pos[7]          = 26D
```

The inspected Teleop `action_dict_to_commands` path creates a
`SingleArmTeleopCommand` from `eef_9d` and `hand_joint_target`, so the 19D
Machine-B action is exactly the command space the robot consumes. Machine A's
seven `arm_joint_target` values remain available as frozen-checkpoint reference
provenance, but they never enter the actor, critic, replay action, action
statistics, fallback command, or hardware converter.

The Machine-B boundary must validate an exact 26D source dimension, full source
layout hash, and rotation convention, then select the declared indices `0:19`.
It independently validates the resulting 19D semantic layout hash. An implicit
slice, padding, or execution-only projection is forbidden.

This changes the actor and critic tensor shapes. Former 26D RLT actor/critic
checkpoints, actor snapshots, action statistics, and replay journals are not
compatible and must not be reused. Start a separate 19D run. The frozen 400k
GR00T checkpoint on Machine A remains compatible because its 26D output is
preserved.

## Rotation contract

The authoritative inference rot6d ordering is row-first:

```text
[r00, r01, r02, r10, r11, r12]
```

The LeRobot v3 bridge reorders state groups from
`arm7 + eef9 + hand10` to Machine A's `eef9 + hand10 + arm7`. It validates the
declared row-first convention and copies the six rotation values without a
transpose. Source-reference and executed-action hashes include this semantic
rotation convention; a dataset declaring any other convention must fail closed.

## Other unresolved live-runtime contracts

- Teleop's base `RobotInterface` currently exposes
  `connect/send_command/stop/disconnect`; it does not define the
  `reset/observe/step` contract expected by a chunk environment. Robot state is
  available through profile-specific trace snapshots, which should first be
  promoted to a stable observation adapter.
- Teleop records terminal success/failure but does not currently expose a
  numeric per-step reward contract. Until configured, RLT should use only an
  explicit sparse terminal label and must not invent dense rewards.
- The existing `rlt_online_rl` `human_override_factory` replaces a whole chunk.
  Teleop can take over in the middle of a chunk and carries intervention IDs,
  resume-gate state, and safety-hold state. A correct adapter must record the
  authority decision at each executed control tick instead of flattening it to
  one chunk-level Boolean.
- The reviewed Teleop policy runs near 10 Hz with a 32-step GR00T horizon and
  an 8-step replan convention, while the current RLT example uses a 50 Hz
  control loop and a 10-step chunk. The execution rate, replan interval,
  sample-and-hold/interpolation rule, and replay timestamps must be made one
  explicit contract.

## Future adapter protocol

The future adapter should structurally implement Teleop's `PolicyInterface`:

```python
class GrootRltTeleopPolicy:
    @property
    def policy_id(self) -> str: ...

    @property
    def policy_version(self) -> str: ...

    def reset(self, *, episode_id: str | None = None) -> None: ...

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk: ...
```

Its constructor/configuration must require, rather than guess:

- Machine-A URL and actor-service URL;
- 26D source-reference, 19D executed-action, 26D proprio, RL-token, and chunk
  dimensions;
- separate ordered hashes for the 26D source reference, 19D action, and 26D
  proprio;
- the exact 26D-to-19D projection indices and
  `groot_row_major_first_two_rows` rot6d convention;
- action representation and versioned normalization-statistics file;
- camera key mapping and task-instruction source;
- command converter/profile name;
- request deadline and a fail-safe result on timeout;
- deterministic/evaluation mode;
- policy/checkpoint/version identifiers written to capture metadata.

Before returning a chunk it must validate:

- exact feature dimensions and finite values;
- exact source-reference, projected-action, and proprio hash equality across
  server, stats, and adapter;
- exact 26D-to-19D projection, with no arm-reference channel reaching the actor,
  critic, replay action, fallback, or command converter;
- row-first rot6d equality without a bridge transpose;
- physical versus normalized action space;
- horizon and action timestamps;
- every action channel is consumed exactly once by the command converter;
- the result passes Teleop's existing rollout guard.

RTC should remain in Teleop's execution/trajectory layer. The GR00T feature
server is intentionally stateless so replay features can be reconstructed out
of order.

## Provenance mapping

The live capture adapter must preserve these distinctions:

| Teleop authority result | RLT meaning |
| --- | --- |
| policy command executed unchanged | VLA or actor behavior, according to the selected policy source |
| human intervention command executed | human-intervention behavior; retain actor and VLA proposals separately |
| safety/hold command | safety hold, not an ordinary human correction |
| policy proposal during dry-run | proposal only; never mark it executed |
| terminal success/failure event | sparse terminal label with its label source |

`source_name` alone is not enough. Store the authority state, intervention ID,
19D executed command, 19D actor proposal, full 26D VLA source reference, its 19D
projected reference, timestamps, projection indices, rotation convention, and
layout/version metadata at each decision boundary.

## Offline exporter caveat

Teleop's current `RltEpisodeExporter` is useful as a schema handoff, but its
exported RL-token tensors are empty and its VLA reference keys are unset. Such
an export is not a ready-to-train VLA warmup replay. It must first be enriched
with RL tokens and complete 26D reference chunks from the exact frozen
checkpoint; the bridge then validates and derives the 19D replay reference.
The canonical LeRobot v3 row-first rot6d values are not transposed during this
process. Validate the result with `groot_rlt.episode_schema` before replay
construction.

## Acceptance gates before enabling hardware

1. Unit-test observation mapping, row-first rot6d preservation, the complete
   26D source layout, and the explicit 19D action projection.
2. Run recorded-episode replay with no hardware output.
3. Run Teleop `rollout_dry_run`; compare candidate, reference, and authority
   logs without sending commands.
4. Run guarded simulation/digital-twin rollout with forced takeover and timeout
   tests.
5. Verify source-reference, projected-action, proprio, and rotation-semantic
   hashes against captured physical actions; use arm-channel sentinels to prove
   the final seven reference values never enter the action path.
6. Verify reset, reward, terminal label, intervention, resume, and safety-hold
   semantics.
7. Verify that no former 26D actor/critic checkpoint, snapshot, statistics file,
   or replay journal is present in the 19D run directory.
8. Only then add an explicit hardware-enabled launch; it must remain opt-in.

Until all gates pass, the repository's existing robot/teleop launchers remain
the only implemented interfaces and no Groot-RLT command selects them by
default.
