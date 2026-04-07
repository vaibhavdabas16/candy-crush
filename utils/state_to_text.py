from __future__ import annotations

from env.candy_env import CandyEnv


SPECIAL_NAMES = {
    CandyEnv.NORMAL: "normal",
    CandyEnv.STRIPED_HORIZONTAL: "striped-horizontal clears its row when swapped or cleared",
    CandyEnv.STRIPED_VERTICAL: "striped-vertical clears its column when swapped or cleared",
    CandyEnv.WRAPPED: "wrapped clears a 3x3 area when swapped or cleared",
    CandyEnv.BLACK: "black clears every candy of the swapped color",
}


def _cell_text(env: CandyEnv, row: int, col: int) -> str:
    value = int(env.board[row, col])
    special = int(env.specials[row, col])
    if special == CandyEnv.STRIPED_HORIZONTAL:
        return f"{value}H"
    if special == CandyEnv.STRIPED_VERTICAL:
        return f"{value}V"
    if special == CandyEnv.WRAPPED:
        return f"{value}W"
    if special == CandyEnv.BLACK:
        return "B*"
    return str(value)


def state_to_text(env: CandyEnv, max_actions: int | None = 20, include_special_rules: bool = True) -> str:
    lines = [
        f"Score: {env.score:.1f}",
        f"Moves left: {env.moves_left}",
        "Board:",
    ]
    lines.extend(
        " ".join(_cell_text(env, row, col) for col in range(env.grid_size))
        for row in range(env.grid_size)
    )
    if include_special_rules:
        present = sorted({int(v) for v in env.specials.reshape(-1) if int(v) != CandyEnv.NORMAL})
        lines.append("Special candy rules:")
        if present:
            for special in present:
                lines.append(f"- {SPECIAL_NAMES[special]}")
        else:
            lines.append("- none on this board")
    valid_actions = env.valid_actions()
    lines.append(f"Valid actions ({len(valid_actions)} total):")
    shown_actions = valid_actions if max_actions is None else valid_actions[:max_actions]
    for action in shown_actions:
        pos_a, pos_b = env.decode_action(action)
        reward = env.simulate_action_reward(action)
        lines.append(f"- {action}: swap {pos_a} {pos_b} immediate_reward={reward:.1f}")
    if max_actions is not None and len(valid_actions) > max_actions:
        lines.append(f"- ... {len(valid_actions) - max_actions} more")
    return "\n".join(lines)
