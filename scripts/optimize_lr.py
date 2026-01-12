#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nothanks_cui import (  # noqa: E402
    FEATURE_NAMES,
    HEURISTIC_V2_PARAMS,
    HeuristicAI,
    LogisticModel,
    NoThanksGame,
    PlayerState,
    extract_features,
)


@dataclass
class EvalResult:
    winrate: float
    mean_score: float


class InlineLogisticAI:
    def __init__(self, name: str, model: LogisticModel):
        self.name = name
        self.model = model

    def choose(self, game: NoThanksGame, idx: int) -> str:
        ps = game.players[idx]
        if ps.tokens == 0:
            return "take"
        feats = extract_features(
            game.players, game.active_card, game.tokens_on_card, game.remaining_draw(), idx
        )
        p_take = self.model.predict_proba(feats)
        return "take" if game.rng.random() < p_take else "pass"


def load_model(path: Path) -> LogisticModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if list(payload.get("feature_names", [])) != list(FEATURE_NAMES):
        raise SystemExit("Feature names mismatch with current code.")
    return LogisticModel(
        feature_names=payload["feature_names"],
        weights=payload["weights"],
        bias=payload["bias"],
        mean=payload["mean"],
        std=payload["std"],
    )


def evaluate(model: LogisticModel, games: int, eval_seed: int) -> EvalResult:
    n = 3
    tokens = NoThanksGame.starting_tokens(n)
    win_count = 0.0
    scores: List[int] = []

    heuristic_agents = [
        HeuristicAI(f"H{i}", params=HEURISTIC_V2_PARAMS) for i in range(n)
    ]

    for seat in range(n):
        seat_rng = random.Random(eval_seed + seat * 10007)
        for _ in range(games):
            players = [PlayerState(f"AI{i}", tokens) for i in range(n)]
            game = NoThanksGame(players, seed=seat_rng.randrange(1 << 30))
            game.setup()

            lr_agent = InlineLogisticAI("LR", model)
            agents = [
                lr_agent if i == seat else heuristic_agents[i] for i in range(n)
            ]

            while not game.is_over():
                idx = game.current
                action = agents[idx].choose(game, idx)
                ps = game.players[idx]
                if action == "pass" and ps.tokens == 0:
                    action = "take"
                if action == "pass":
                    game.apply_pass()
                else:
                    game.apply_take()

            game_scores = game.provisional_scores()
            scores.append(game_scores[seat])
            best = min(game_scores)
            winners = [i for i, s in enumerate(game_scores) if s == best]
            if seat in winners:
                win_count += 1.0 / len(winners)

    total_games = games * n
    return EvalResult(
        winrate=win_count / total_games,
        mean_score=sum(scores) / len(scores),
    )


def is_better(candidate: EvalResult, best: EvalResult) -> bool:
    if candidate.winrate > best.winrate:
        return True
    if abs(candidate.winrate - best.winrate) < 1e-9:
        return candidate.mean_score < best.mean_score
    return False


def save_model(
    out_path: Path,
    base_path: Path,
    model: LogisticModel,
    meta: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "weights": model.weights,
        "bias": model.bias,
        "mean": model.mean,
        "std": model.std,
        "meta": {"base_model": str(base_path), **meta},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-search optimization for LR weights")
    parser.add_argument("--base", required=True, help="Base model path (json)")
    parser.add_argument("--out", required=True, help="Output model path (json)")
    parser.add_argument("--games", type=int, default=200, help="Games per seat")
    parser.add_argument("--iters", type=int, default=80, help="Search iterations")
    parser.add_argument("--sigma", type=float, default=0.15, help="Stddev for weights")
    parser.add_argument("--sigma-bias", type=float, default=0.15, help="Stddev for bias")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for search")
    parser.add_argument("--eval-seed", type=int, default=1, help="RNG seed for eval")
    parser.add_argument("--report-every", type=int, default=10)
    args = parser.parse_args()

    base_path = Path(args.base)
    out_path = Path(args.out)

    base_model = load_model(base_path)
    base_eval = evaluate(base_model, games=args.games, eval_seed=args.eval_seed)
    print(
        "BASE",
        f"winrate={base_eval.winrate:.2%}",
        f"mean_score={base_eval.mean_score:.2f}",
    )

    best_model = LogisticModel(
        feature_names=base_model.feature_names,
        weights=list(base_model.weights),
        bias=base_model.bias,
        mean=list(base_model.mean),
        std=list(base_model.std),
    )
    best_eval = base_eval

    rng = random.Random(args.seed)

    for i in range(1, args.iters + 1):
        cand_weights = [w + rng.gauss(0, args.sigma) for w in best_model.weights]
        cand_bias = best_model.bias + rng.gauss(0, args.sigma_bias)
        cand_model = LogisticModel(
            feature_names=best_model.feature_names,
            weights=cand_weights,
            bias=cand_bias,
            mean=best_model.mean,
            std=best_model.std,
        )
        cand_eval = evaluate(cand_model, games=args.games, eval_seed=args.eval_seed)
        if is_better(cand_eval, best_eval):
            best_model = cand_model
            best_eval = cand_eval
            print(
                f"NEWBEST i={i} winrate={best_eval.winrate:.2%} mean_score={best_eval.mean_score:.2f}"
            )

        if args.report_every and i % args.report_every == 0:
            print(
                f"iter={i} best_winrate={best_eval.winrate:.2%} best_mean_score={best_eval.mean_score:.2f}"
            )

    meta = {
        "method": "random_hill_climb",
        "games_per_seat": args.games,
        "iters": args.iters,
        "sigma": args.sigma,
        "sigma_bias": args.sigma_bias,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "base_winrate": base_eval.winrate,
        "base_mean_score": base_eval.mean_score,
        "best_winrate": best_eval.winrate,
        "best_mean_score": best_eval.mean_score,
    }
    save_model(out_path, base_path, best_model, meta)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
