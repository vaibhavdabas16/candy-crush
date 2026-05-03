# Qwen GRPO Evaluation

This document summarizes the released Qwen GRPO LoRA adapter and how it compares with the other strategies in this repo.

## Released Adapter

- Base model: `Qwen/Qwen3.5-9B`
- LoRA adapter: `arnavm7/candy-crush-qwen35-grpo-lora`
- GUI agent: `llm_grpo`
- Inference output: text command, for example `swap (3,5) (3,6)`
- Training method: GRPO with LoRA/QLoRA, bf16 compute, `beta_kl=0`

The adapter is loaded by the GUI either from Hugging Face directly:

```bash
python run_gui.py --agent llm_grpo \
  --llm-grpo-path arnavm7/candy-crush-qwen35-grpo-lora \
  --llm-model-name Qwen/Qwen3.5-9B
```

or from a local adapter directory downloaded with `huggingface_hub.snapshot_download`.

## Fixed One-Swap Eval

The final comparison was run on a fixed set of 10 special-candy boards. Each policy receives the same board state and is scored on the immediate reward of one selected swap. Higher is better.

This is not a full-episode score. It answers a narrower question: "given this exact board, how good is the next swap?"

```text
avg reward = mean(simulate_action_reward(policy_chosen_action) over the fixed eval boards)
```

The fixed board seeds were:

```text
20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009
```

Special-candy injection is deterministic too. For board seed `s`, the eval uses `special_seed = s + 50000`, so every policy sees the same normal candies and the same injected specials.

Eval environment:

| Setting | Value |
| --- | --- |
| Board size | `8x8` |
| Normal colors | `0..5` |
| Action space | Adjacent swaps only, 112 possible swaps |
| Eval horizon | One selected swap per board |
| Specials per eval board | 1 to 3 injected specials |
| Invalid action score | `-5` |

| Policy | Avg reward | Notes |
| --- | ---: | --- |
| Greedy | 337.9 | Searches all legal swaps and chooses the highest immediate simulated reward. |
| Qwen GRPO LoRA `final_plus30` | 302.0 | Text policy trained with GRPO, then validated through the same `CandyEnv` legality/reward path. |
| Qwen GRPO LoRA `final` | 295.5 | Earlier final checkpoint before the extra 30-step continuation. |
| Random | 146.0 | Uniform valid random swap. |
| PPO | 125.9 | PPO checkpoint from the repo's baseline training flow. |
| DQN | 90.4 | DQN checkpoint from the repo's baseline training flow. |

The released adapter is the `final_plus30` checkpoint. It beat random, PPO, and DQN on this fixed eval, and landed close to the greedy immediate-reward baseline.

## Reward Function

The reward is computed by the real `CandyEnv` cascade resolver.

For a normal match/cascade clear:

```text
reward += removed_candies^2 + 10
```

For a direct special activation before normal cascades continue:

```text
reward += removed_candies^2 + special_bonus
```

Special bonuses are:

| Special | Bonus |
| --- | ---: |
| Horizontal striped | 20 |
| Vertical striped | 20 |
| Wrapped | 40 |
| Black/color-bomb | 60 |

After each clear, gravity and refill run, then the environment keeps resolving cascades until no match remains. That means a swap creating a large clear or triggering specials can score much more than a basic 3-candy match.

Examples:

| Event | Reward contribution |
| --- | ---: |
| Basic 3-candy normal match | `3^2 + 10 = 19` |
| 4-candy normal match | `4^2 + 10 = 26` |
| 5-candy normal match | `5^2 + 10 = 35` |
| Horizontal stripe clears 8 candies directly | `8^2 + 20 = 84` |
| Wrapped special clears a 3x3 area directly | `9^2 + 40 = 121` |

`simulate_action_reward(action)` temporarily applies the swap, resolves those clears/cascades, returns the reward, and then restores the original board state. Greedy uses this for every legal action; the Qwen prompt also includes immediate simulated rewards in the legal-action list.

## How The Qwen Policy Works

The GUI passes the current `CandyEnv` state to `LLMGRPOAgent`.

1. The board is serialized to text with coordinates, normal candy colors, special candy markers, remaining moves, legal swaps, immediate simulated rewards, and the known special-candy rules.
2. Qwen plus the LoRA adapter generates text.
3. The parser extracts a command matching `swap (r,c) (r,c)`.
4. The parsed action is checked against `env.is_valid_action`.
5. If parsing or legality fails, the agent falls back to the best immediate legal swap, so the GUI still receives a valid move.

The current board encoding supports:

| Encoding | Meaning |
| --- | --- |
| `0` through `5` | Normal candy colors. |
| `3H` | Color 3 horizontal striped candy. |
| `3V` | Color 3 vertical striped candy. |
| `3W` | Color 3 wrapped candy. |
| `B*` | Black/color-bomb candy. |

## Why Greedy Is Still Strong

Greedy has direct access to `env.simulate_action_reward` for every legal swap, so it is a strong one-step oracle. On an immediate one-swap eval, greedy should usually be the ceiling unless a learned policy captures longer-term value that is not represented by the immediate reward.

The Qwen GRPO policy is useful because it maps a text board state to a text recommendation, can consume rule text in the prompt, and can be integrated into UI/API workflows that expect natural-language or command-like output.

## What Happens If A New Candy Type Is Added

The expected winner depends on what is updated.

| Scenario | Expected behavior |
| --- | --- |
| New candy is added to the board text only, but `CandyEnv`, legal moves, and rewards are not updated | No policy can truly use it. The model may mention it, but the environment will not score it correctly. |
| `CandyEnv` and `state_to_text` are updated with the new candy rule, and the prompt explains the rule | Qwen GRPO has the best zero-shot chance among the learned policies because it can read the new rule text. Greedy will also handle it well if `simulate_action_reward` knows the new rule. |
| DQN/PPO are evaluated on the new candy without retraining | They will usually degrade because their observation distribution changed and they do not read textual rules. |
| DQN/PPO are retrained with the new candy | They can recover, but they need enough new rollouts and may still be less flexible for future rule edits. |
| The new candy introduces a new action format beyond adjacent swaps | The parser, action space, GUI, and all policies need code changes. The current Qwen agent only returns adjacent `swap (r,c) (r,c)` commands. |

For a new candy type that still uses adjacent swaps, the practical order is usually:

1. Greedy, if the simulator knows the new candy and the objective is immediate reward.
2. Qwen GRPO, if the prompt describes the new candy and the environment validates/rewards it.
3. Retrained PPO/DQN after collecting enough new training experience.
4. Old PPO/DQN without retraining.

For longer-horizon play, the right test is a full-episode eval, not the one-swap snapshot above. Greedy can miss setup moves; a learned policy can beat it only if the reward/eval protocol values future cascades and the model was trained for that horizon.

## Re-running The Eval

Baseline one-step comparisons are produced by the Qwen GRPO trainer path:

```bash
python train/train_llm_grpo_candy.py \
  --model-name Qwen/Qwen3.5-9B \
  --run-dir models/llm_grpo_candy/qwen35_9b \
  --skip-baseline-training \
  --eval-episodes 10 \
  --eval-seed 20000
```

To evaluate a specific adapter on fixed seeds:

```bash
python train/train_llm_grpo_candy.py \
  --eval-adapter models/llm_grpo_candy/qwen35_9b/final_plus30 \
  --model-name Qwen/Qwen3.5-9B \
  --eval-seeds-csv 20000,20001,20002,20003,20004,20005,20006,20007,20008,20009
```

The normal repo eval script still covers full-episode random, greedy, DQN, and PPO:

```bash
python eval/evaluate.py
```
