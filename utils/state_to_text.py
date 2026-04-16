from __future__ import annotations

from env.candy_env import CandyEnv


def state_to_text(env: CandyEnv, max_actions: int = 20) -> str:
    lines = [
        f"Score: {env.score:.1f}",
        f"Moves left: {env.moves_left}",
        "Board:",
    ]
    lines.extend(" ".join(str(int(value)) for value in row) for row in env.board)
    valid_actions = env.valid_actions()
    lines.append(f"Valid actions ({len(valid_actions)} total):")
    for action in valid_actions[:max_actions]:
        pos_a, pos_b = env.decode_action(action)
        lines.append(f"- {action}: swap {pos_a} with {pos_b}")
    if len(valid_actions) > max_actions:
        lines.append(f"- ... {len(valid_actions) - max_actions} more")
    return "\n".join(lines)
