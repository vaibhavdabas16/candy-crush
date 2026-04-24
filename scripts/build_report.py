"""Build the 6-page IEEE-style RL report from the team template.

Reads the template at ../RL report Template.docx (relative to where this
is invoked) and writes report/RL_Report.docx with our content plugged
into the template's IEEE styles.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.shared import Pt, Inches

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #

ABSTRACT = (
    "We present a comparative study of reinforcement-learning policies on "
    "an 8x8 Candy Crush variant: a stochastic match-three grid puzzle with "
    "112 swap actions, cascading rewards, and four classes of special "
    "candies. We benchmark Random and Greedy baselines, a Deep Q-Network "
    "(DQN) with action masking, Proximal Policy Optimization (PPO) with a "
    "maskable action head, and a 9-billion-parameter Qwen language model "
    "fine-tuned with Group Relative Policy Optimization (GRPO) and served "
    "as a quantized Q4_K_M GGUF for CPU and Metal inference. All five "
    "policies are evaluated on a fixed suite of ten special-candy boards "
    "with full 20-move rollouts. The GRPO LLM policy achieves a 99.5 "
    "percent legal-action rate without any rule-based fallback and "
    "outperforms DQN and PPO on seven of ten boards. A one-step Greedy "
    "oracle remains the strongest single-decision policy on average. We "
    "release the entire pipeline, including a one-command Ubuntu 22.04 "
    "reproduction script."
)

KEYWORDS = None  # keywords paragraph is deleted

# (heading_level_or_para_style, text)  -- heading_level == "h1"|"h2"|"h3"|"p"|"bullet"
SECTIONS = [
    ("h1", "Introduction and Problem Statement"),
    (
        "p",
        "Match-three puzzle games such as Candy Crush Saga combine simple "
        "local rules with a high-dimensional combinatorial action space "
        "and stochastic re-fill dynamics that make long-horizon planning "
        "difficult. A human player on an 8x8 board chooses a single swap "
        "between two adjacent cells, but the resulting reward depends on "
        "match clearings, gravity, refills, and recursive cascades that "
        "can be triggered by special candies. The agent therefore "
        "operates in an environment where a single action induces a "
        "complex, partially stochastic chain of consequences whose value "
        "is hard to capture with shallow features.",
    ),
    (
        "p",
        "We study this domain as a reinforcement-learning testbed. Our "
        "concrete questions are (1) how strong is a one-step Greedy "
        "oracle that has direct access to a deterministic forward "
        "simulator, (2) whether off-the-shelf deep RL agents (DQN, PPO) "
        "can compete with or beat that oracle on full episodes, and (3) "
        "whether a large language model fine-tuned with Group Relative "
        "Policy Optimization (GRPO) can be turned into a competitive, "
        "rule-aware policy that is auditable through natural-language "
        "reasoning. The third question is motivated by the observation "
        "that a textual representation of the board is information-"
        "preserving, and that an LLM can be taught to emit a single "
        "swap command that the environment then validates and rewards.",
    ),
    (
        "p",
        "We released the full system as a single bash pipeline that "
        "trains the deep-RL baselines from scratch, evaluates every "
        "policy on a shared set of seeds, and integrates a 9-billion-"
        "parameter Qwen GRPO model via a 5.3 GB merged GGUF for fast "
        "non-CUDA inference. Our evaluation protocol fixes ten special-"
        "candy boards (seeds 20000-20009 with deterministic special-"
        "candy injection at seed+50000) and plays every policy through a "
        "complete 20-move rollout, so the compared metric is total "
        "episode reward rather than a one-step decision score.",
    ),

    ("h1", "RL Formulation"),

    ("h2", "A. Environment"),
    (
        "p",
        "The environment is implemented as a Gymnasium-compatible "
        "CandyEnv on an 8x8 grid with six normal candy colors and four "
        "special-candy types: horizontal striped, vertical striped, "
        "wrapped, and color-bomb. The board state is therefore a pair of "
        "8x8 integer arrays, one storing the candy color and one storing "
        "the special-candy type, plus a scalar count of remaining moves.",
    ),

    ("h2", "B. State Representation"),
    (
        "p",
        "For the deep-RL agents the observation is a 65-dimensional "
        "float32 vector: a flattened 8x8 grid of normalized color codes "
        "plus the normalized number of moves remaining. The action space "
        "is a Discrete(112) over all adjacent swaps; the env exposes an "
        "is_valid_action(a) predicate and an action_masks() helper that "
        "DQN and Maskable PPO use to restrict action selection. For the "
        "LLM agent we serialize the same state to a text prompt that "
        "lists the board grid, the special-candy markers, the remaining "
        "moves, and the legal swaps with their immediate simulated "
        "reward.",
    ),

    ("h2", "C. Action Space"),
    (
        "p",
        "There are 7x8 = 56 horizontal swaps and 8x7 = 56 vertical "
        "swaps for a total of 112 actions. Each action a is decoded into "
        "a pair of grid positions ((r1,c1),(r2,c2)). An action is legal "
        "iff applying the swap creates at least one match-of-three or "
        "activates a special candy.",
    ),

    ("h2", "D. Reward Model"),
    (
        "p",
        "After a legal swap, the environment runs a deterministic "
        "cascade resolver. For every cleared group of n candies the step "
        "receives n^2 + 10 points (n^2 + special_bonus when a special "
        "candy is consumed, with bonuses 20 / 20 / 40 / 60 for the four "
        "special types). Cascades repeat until the board stabilizes, so "
        "a single swap can yield several hundred points. An illegal "
        "swap is penalized with reward -5 and consumes a move. The "
        "episode terminates after max_moves=20 swaps. This reward "
        "scaling is dense and approximately quadratic in the size of "
        "the cleared group, which heavily rewards setup moves that "
        "trigger long chains.",
    ),

    ("h2", "E. MDP Structure"),
    (
        "p",
        "The decision process is a finite-horizon Markov decision "
        "process with discrete actions, a deterministic match resolver, "
        "and a stochastic refill step that samples new candies "
        "uniformly from the six normal colors. Because the refill "
        "introduces fresh randomness each step, the same action from "
        "the same observed state can yield different next states. The "
        "objective is the undiscounted total return over the 20-move "
        "horizon, although DQN and PPO are trained with gamma=0.9 to "
        "smooth value estimates.",
    ),
    (
        "p",
        "Because legal moves are computed exactly by the simulator, "
        "every learning algorithm in this study uses action masking to "
        "block the trivial -5 self-penalty path; the LLM policy in "
        "evaluation has masking disabled so that any parse failure or "
        "illegal swap is counted honestly against it.",
    ),

    # --- Page 3-4 ---

    ("h1", "Methodology"),

    ("h2", "A. Greedy and Random Baselines"),
    (
        "p",
        "Random uniformly samples a legal swap each step. Greedy is a "
        "one-step oracle: it queries env.simulate_action_reward(a) for "
        "every legal action a, picks the maximum, and breaks ties "
        "uniformly. Greedy has direct access to the deterministic match "
        "resolver and is therefore an extremely strong single-decision "
        "policy in this environment.",
    ),

    ("h2", "B. Deep Q-Network (DQN)"),
    (
        "p",
        "We train a DQN with a 256-wide, 2-layer MLP Q-head, replay "
        "buffer size 50000, target-network updates every 500 steps, "
        "epsilon-greedy exploration with a 15000-step linear decay, "
        "Huber loss, and Adam (lr=1e-3). At action selection we mask "
        "Q-values of illegal actions to -infinity. Training is logged to "
        "TensorBoard along with mean episode reward, invalid-action "
        "rate, and a moving-average return over a 20-episode window.",
    ),

    ("h2", "C. Proximal Policy Optimization (PPO)"),
    (
        "p",
        "PPO uses Stable-Baselines3 with sb3-contrib's MaskablePPO when "
        "available, otherwise vanilla PPO with manual masking applied "
        "before action sampling. The default actor-critic MLP is two "
        "256-wide hidden layers; gamma is 0.9 and rollouts are 2048 "
        "steps. Training time is wall-clock-capped to roughly five "
        "minutes inside the bash pipeline; longer runs are exposed via "
        "the PPO_TIMESTEPS environment variable.",
    ),

    ("h2", "D. GRPO Language-Model Policy"),
    (
        "p",
        "The LLM policy is a Qwen-3.5 9B base model fine-tuned with "
        "Group Relative Policy Optimization (GRPO) using a LoRA adapter "
        "of rank 64 on the attention projections. The training reward "
        "is the env's reward for the swap parsed out of the model's "
        "generation, so the policy is trained directly against the "
        "ground-truth game simulator without imitation data. After "
        "training, the LoRA adapter is merged back into the base "
        "weights and the merged checkpoint is quantized to Q4_K_M GGUF "
        "(5.3 GB), which lets us run inference on a CPU-only Ubuntu "
        "container or with full Metal/CUDA offload through llama-cpp-"
        "python.",
    ),
    (
        "p",
        "At inference time we serialize the board state to a text "
        "prompt that includes the legal-action list with simulated "
        "rewards, then ask the model for a single line of the form "
        "'swap (r,c) (r,c)' followed by an optional one-line reason. "
        "We run greedy decoding with temperature 0 and a 24-token cap. "
        "A regex parser extracts the swap; if either the parse or the "
        "is_valid_action check fails, the agent records a parse "
        "failure or model-invalid event and (in eval mode) emits a "
        "deliberately illegal action so the env applies the -5 "
        "penalty. This 'no_fallback' contract makes the LLM's measured "
        "validity a real property of the model rather than an artifact "
        "of a rule-based safety net.",
    ),

    ("h2", "E. End-to-End Pipeline"),
    (
        "p",
        "The deliverable is a single bash script, run.sh, that "
        "executes a seven-stage pipeline in a fresh Ubuntu 22.04 "
        "container with no manual configuration: (1) apt-install build "
        "tools, (2) create a venv and install the baseline plus "
        "GRPO-inference dependencies, (3) train DQN, (4) train PPO, "
        "(5) play Random/Greedy/DQN/PPO on shared seeds and print a "
        "comparison table, (6) download the merged Q4_K_M GGUF from "
        "Hugging Face and play the GRPO model on the same seeds, and "
        "(7) install the heavy GRPO-training dependencies and run up "
        "to one hour of LoRA GRPO training. The pipeline only uses "
        "relative paths and writes all artifacts under ./logs and "
        "./models so it can be cloned and run in any working "
        "directory.",
    ),

    ("h1", "Contributions"),
    (
        "p",
        "All four authors contributed equally and the work was split "
        "approximately as follows.",
    ),
    # contribution table inserted programmatically below

    # --- Page 5-6 ---

    ("h1", "Experimental Setup"),
    (
        "p",
        "The fixed evaluation protocol uses ten boards generated by "
        "deterministic seeds 20000 through 20009, with special candies "
        "injected from seed+50000. Each rollout lasts 20 moves and "
        "every policy sees the same starting board. DQN, PPO, and the "
        "two analytical baselines run on CPU; the GRPO model is run "
        "via llama-cpp-python with all layers offloaded to Metal on "
        "Apple Silicon (and is similarly compatible with CUDA). All "
        "metrics are computed from the env's own reward signal: "
        "average and standard deviation of the total episode reward, "
        "minimum and maximum board reward, fraction of moves marked as "
        "invalid by the env, and (for GRPO only) the parse-invalid and "
        "model-invalid rates separated.",
    ),

    ("h1", "Results"),

    # results table inserted programmatically below

    (
        "p",
        "Greedy is the strongest policy on average (avg 2314.6, std "
        "548.5), confirming that direct access to the deterministic "
        "forward simulator is hard to beat in single-decision quality. "
        "The GRPO LLM policy is second (avg 1827.8, std 569.2) and "
        "beats DQN and PPO on seven of ten boards; on board 20008 it "
        "wins outright (3005 vs 2940). DQN and PPO at the modest "
        "training budgets used by the bash pipeline land near the "
        "Random baseline, which is consistent with the observation "
        "that the cascade structure of the reward makes credit "
        "assignment hard for short, value-bootstrapped agents.",
    ),
    (
        "p",
        "The most striking property of the GRPO policy is its "
        "validity. Across 200 model decisions in the no-fallback eval, "
        "the parser failed zero times and the model proposed an "
        "illegal swap once, for a 99.5 percent legal-action rate from "
        "the LLM alone. The GRPO training reward, which only credits "
        "the model when a parsed swap is both well-formed and legal, "
        "appears to have collapsed the failure modes of free-form "
        "generation almost entirely. This is also visible in the "
        "raw-output trace: the model often emits a 'Reason:' line "
        "after the swap command, suggesting that the policy has "
        "internalized a small explanation step.",
    ),
    (
        "p",
        "Two limitations are worth flagging. First, the deep RL "
        "baselines were trained for only a few minutes inside the "
        "bash pipeline, so the comparison should not be read as a "
        "ceiling result for DQN or PPO; longer runs (several hours of "
        "training) typically lift those baselines well above Random. "
        "Second, the Greedy oracle benefits from the dense reward "
        "structure of this environment; in domains with sparser "
        "rewards or longer planning horizons we would expect the "
        "learned policies to gain ground more clearly.",
    ),

    ("h1", "Conclusion"),
    (
        "p",
        "We have presented a complete, reproducible study of "
        "reinforcement-learning policies on an 8x8 Candy Crush "
        "variant. The headline finding is that a 9B-parameter "
        "language model trained with GRPO against the game simulator "
        "is competitive with classical deep-RL baselines while also "
        "being auditable: every decision is a human-readable swap "
        "command and an optional rationale, and the model's legal-"
        "action rate is high enough that the entire safety net of "
        "rule-based fallbacks can be removed without collapsing "
        "performance. We release the model on Hugging Face and the "
        "full pipeline as a single bash script that runs end-to-end "
        "in a fresh Ubuntu 22.04 container.",
    ),

    ("h1", "References"),
    ("ref", "[1] J. Schulman et al., 'Proximal Policy Optimization Algorithms,' arXiv:1707.06347, 2017."),
    ("ref", "[2] V. Mnih et al., 'Human-level control through deep reinforcement learning,' Nature, vol. 518, 2015."),
    ("ref", "[3] Z. Shao et al., 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models,' arXiv:2402.03300, 2024. (GRPO formulation)"),
    ("ref", "[4] E. J. Hu et al., 'LoRA: Low-Rank Adaptation of Large Language Models,' arXiv:2106.09685, 2021."),
    ("ref", "[5] G. Gerganov, 'llama.cpp: LLM inference in C/C++,' https://github.com/ggml-org/llama.cpp, 2023."),
    ("ref", "[6] Qwen Team, 'Qwen3.5 Technical Report,' Hugging Face, 2025."),
    ("ref", "[7] A. Raffin et al., 'Stable-Baselines3,' Journal of Machine Learning Research, vol. 22, 2021."),
]


CONTRIBUTIONS = [
    ("Ananya Singla",
     "Designed and implemented the CandyEnv environment (board generation, special-candy "
     "rules, cascade resolver, action encoding). Authored the random and greedy baselines "
     "and the Pygame visual viewer used during development."),
    ("Arnav Mehta",
     "Proposed and led the novel idea of training a 9B language model directly against the "
     "game simulator with GRPO. Implemented the LoRA fine-tuning of Qwen 3.5-9B, the prompt "
     "and reward design for textual swap commands, the Q4_K_M GGUF merge, the llama-cpp "
     "inference agent (LLMGRPOGGUFAgent), and the fixed-board evaluation harness."),
    ("Mohil Ahuja",
     "Built the textual board serialization, special-candy rule encoding, and prompt "
     "template that the LLM consumes. Implemented the parser and validity-check pipeline "
     "(parse-failure / model-invalid bookkeeping) and the no-fallback eval contract."),
    ("Vaibhav Dabas",
     "Authored the DQN and PPO training scripts, the saved-model loader, and the seven-"
     "stage run.sh pipeline that automates the entire workflow on a fresh Ubuntu 22.04 "
     "container, including Docker validation."),
]


RESULTS_TABLE = [
    ("Policy", "Avg ± Std", "Min", "Max", "Invalid"),
    ("Random",    "1072.9 ± 218.1", "697",  "1383", "0.000"),
    ("PPO",       "978.5 ± 273.6",  "583",  "1460", "0.000"),
    ("DQN",       "1092.8 ± 233.3", "668",  "1596", "0.000"),
    ("GRPO GGUF", "1827.8 ± 569.2", "1193", "3005", "0.005"),
    ("Greedy",    "2314.6 ± 548.5", "1511", "3100", "0.000"),
]

PER_BOARD_TABLE = [
    ("Seed", "Random", "PPO", "DQN", "GRPO", "Greedy"),
    ("20000", "1383", "781",  "979",  "1258", "1511"),
    ("20001", "1266", "808",  "1210", "1193", "2486"),
    ("20002", "697",  "835",  "1142", "2371", "2387"),
    ("20003", "854",  "583",  "1156", "1872", "1916"),
    ("20004", "1026", "1460", "906",  "2209", "1813"),
    ("20005", "958",  "1378", "956",  "1810", "2206"),
    ("20006", "1371", "1014", "1596", "2034", "3100"),
    ("20007", "983",  "715",  "668",  "1224", "1728"),
    ("20008", "955",  "1038", "1069", "3005", "2940"),
    ("20009", "1236", "1173", "1246", "1302", "3059"),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def remove_paragraph(p):
    el = p._element
    el.getparent().remove(el)


def style(name, doc):
    return doc.styles[name]


def add_para(doc, text, style_name):
    p = doc.add_paragraph(text, style=style_name)
    return p


def add_table_with_style(doc, rows, header_style="IEEE Table Header Centered",
                        cell_style="IEEE Table Cell"):
    cols = len(rows[0])
    t = doc.add_table(rows=len(rows), cols=cols)
    t.style = "Table Grid"
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            try:
                p.style = doc.styles[header_style if ri == 0 else cell_style]
            except KeyError:
                pass
            if ri == 0:
                run.bold = True
    return t


def emit_section(doc, items):
    for kind, text in items:
        if kind == "h1":
            add_para(doc, text, "IEEE Heading 1")
        elif kind == "h2":
            add_para(doc, text, "IEEE Heading 2")
        elif kind == "h3":
            add_para(doc, text, "IEEE Heading 3")
        elif kind == "p":
            add_para(doc, text, "IEEE Paragraph")
        elif kind == "ref":
            add_para(doc, text, "IEEE Reference Item")
        elif kind == "bullet":
            add_para(doc, text, "IEEE Bullet 1")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(template: Path, output: Path) -> None:
    doc = Document(str(template))

    # 1) Replace abstract + keywords. The abstract paragraph is detected by
    #    the IEEE Abtract style; the keywords paragraph is the one that
    #    starts with "Keywords".
    abstract_replaced = False
    keywords_replaced = False
    title_replaced = False
    keywords_para = None
    for p in doc.paragraphs:
        if not title_replaced and p.style.name == "IEEE Title" and p.text.strip():
            # Title spans two paragraphs in the template ("Learning..." +
            # "Grid-Based Puzzles..."). Use the first for the project
            # name and the second as a one-line subtitle.
            p.text = "Candy Crusher"
            title_replaced = True
            continue
        if title_replaced and p.style.name == "IEEE Title" and p.text.strip():
            p.text = "Classical RL Baselines and a GRPO-Trained LLM Policy"
            continue
        if not abstract_replaced and p.style.name == "IEEE Abtract":
            # Keep the leading "Abstract—" run intact, replace the body text.
            new_text = "Abstract—" + ABSTRACT
            p.text = ""
            run = p.add_run(new_text)
            run.bold = False
            abstract_replaced = True
            continue
        if not keywords_replaced and p.text.strip().startswith("Keywords"):
            # Drop the keywords line entirely.
            keywords_para = p
            p.text = ""
            keywords_replaced = True
            continue

    # 2) Drop everything after the keywords paragraph so we can rewrite
    #    the body cleanly.
    paragraphs = list(doc.paragraphs)
    if keywords_para is None:
        raise RuntimeError("Could not locate Keywords paragraph in template.")
    kw_el = keywords_para._element
    keywords_idx = next(
        i for i, p in enumerate(paragraphs) if p._element is kw_el
    )

    # Drop the (now-empty) keywords paragraph itself plus everything
    # after it, and remove every existing table from the body.
    for p in paragraphs[keywords_idx:]:
        remove_paragraph(p)
    for t in list(doc.tables):
        t._element.getparent().remove(t._element)

    # 3) Emit the report body.
    # ---- Section 1 + 2: Problem statement + RL formulation ----
    intro_end = next(i for i, (k, _) in enumerate(SECTIONS) if k == "h1" and _ == "Methodology")
    emit_section(doc, SECTIONS[:intro_end])

    # ---- Section 3: Methodology + Contributions ----
    contrib_start = next(
        i for i, (k, t) in enumerate(SECTIONS) if k == "h1" and t == "Contributions"
    )
    exp_setup_start = next(
        i for i, (k, t) in enumerate(SECTIONS) if k == "h1" and t == "Experimental Setup"
    )
    emit_section(doc, SECTIONS[intro_end:contrib_start])  # Methodology
    emit_section(doc, SECTIONS[contrib_start:exp_setup_start])  # Contributions intro

    # Contributions table (4 rows, equal split)
    contrib_rows = [("Author", "Contribution")] + CONTRIBUTIONS
    add_table_with_style(doc, contrib_rows)

    # ---- Section 4: Results ----
    emit_section(doc, SECTIONS[exp_setup_start:])

    # The Results table goes right after the "Results" heading. Find it
    # and insert the two result tables inline. The cleanest way given
    # python-docx's API: locate the first paragraph whose text starts
    # with "Greedy is the strongest policy" and insert tables BEFORE it.
    insert_anchor = None
    for p in doc.paragraphs:
        if p.text.startswith("Greedy is the strongest policy"):
            insert_anchor = p
            break
    if insert_anchor is None:
        raise RuntimeError("Could not find Results discussion paragraph.")

    # Build the tables at end of doc, then move them to the anchor.
    summary_caption = add_para(
        doc,
        "Table I. Aggregate reward across 10 boards (full 20-move rollouts).",
        "IEEE Table Caption",
    )
    summary_table = add_table_with_style(doc, RESULTS_TABLE)
    perboard_caption = add_para(
        doc,
        "Table II. Per-board total reward by policy.",
        "IEEE Table Caption",
    )
    perboard_table = add_table_with_style(doc, PER_BOARD_TABLE)

    anchor_el = insert_anchor._element
    for el in (
        summary_caption._element,
        summary_table._element,
        perboard_caption._element,
        perboard_table._element,
    ):
        anchor_el.addprevious(el)

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output))
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--template",
        default="/Users/arnavmehta/Downloads/RL report Template.docx",
        help="Path to the IEEE template .docx",
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "report" / "RL_Report.docx"),
        help="Where to write the generated report",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.template), Path(args.output))
