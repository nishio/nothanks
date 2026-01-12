#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nothanks_cui import (  # noqa: E402
    FEATURE_NAMES,
    HEURISTIC_V2_PARAMS,
    HeuristicAI,
    NoThanksGame,
    PlayerState,
    extract_features,
)


HEADER_RE = re.compile(
    r"^\[(?P<card>-|\d+)\]-(?P<tok>\d+)\s+\(Active card: (?P=card)\s+Tokens on card: (?P=tok)\s+Deck left: (?P<deck>\d+)\)"
)
PLAYER_RE = re.compile(
    r"^(?P<turn>[ >])\[(?P<idx>\d+)\]\s+(?P<name>\S+)\s+tokens=\s*(?P<tokens>\d+)\s+cards=(?P<cards>.+?)\s+cardPts=\s*(?P<cardpts>-?\d+)\s+score=\s*(?P<score>-?\d+)"
)
MOVE_RE = re.compile(r"^Your move:.*?>\s*(?P<input>.*)$")
WINNER_RE = re.compile(r"^Winner:\s+(?P<winner>.+)$")
WINNERS_RE = re.compile(r"^Winners:\s+(?P<winners>.+)$")


def parse_cards(cards_str: str) -> set[int]:
    s = cards_str.strip()
    if s == "(none)":
        return set()
    parts = [p.strip() for p in s.split("|")]
    out: set[int] = set()
    for p in parts:
        if "-" in p:
            a, b = p.split("-")
            for x in range(int(a), int(b) + 1):
                out.add(x)
        else:
            out.add(int(p))
    return out


def parse_winners(lines: List[str]) -> set[str]:
    for line in reversed(lines):
        m = WINNER_RE.match(line)
        if m:
            return {m.group("winner").strip()}
        m = WINNERS_RE.match(line)
        if m:
            return {w.strip() for w in m.group("winners").split(",")}
    return set()


@dataclass
class Sample:
    features: List[float]
    label: int
    you_won: bool
    log_name: str


def make_game_state(
    players: List[PlayerState],
    active_card: Optional[int],
    tokens_on_card: int,
    deck_left: int,
    current_idx: int,
) -> NoThanksGame:
    game = NoThanksGame(players, seed=0)
    game.active_card = active_card
    game.tokens_on_card = tokens_on_card
    game.deck = [0] * deck_left
    game.current = current_idx
    return game


def parse_log(path: Path) -> Tuple[List[Sample], dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    winners = parse_winners(lines)
    you_won = "YOU" in winners

    samples: List[Sample] = []
    players_by_idx: dict[int, PlayerState] = {}
    you_idx: Optional[int] = None

    active_card: Optional[int] = None
    tokens_on_card = 0
    deck_left = 0

    baseline = {
        "heuristic": {"match": 0, "total": 0},
        "heuristic2": {"match": 0, "total": 0},
    }
    heuristic = HeuristicAI("H1")
    heuristic2 = HeuristicAI("H2", params=HEURISTIC_V2_PARAMS)

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            active_card = None if m.group("card") == "-" else int(m.group("card"))
            tokens_on_card = int(m.group("tok"))
            deck_left = int(m.group("deck"))
            continue

        m = PLAYER_RE.match(line)
        if m:
            idx = int(m.group("idx"))
            name = m.group("name")
            tokens = int(m.group("tokens"))
            cards = parse_cards(m.group("cards"))
            players_by_idx[idx] = PlayerState(name=name, tokens=tokens, cards=cards)
            if name == "YOU":
                you_idx = idx
            continue

        m = MOVE_RE.match(line)
        if m:
            if you_idx is None:
                continue
            if not players_by_idx:
                continue
            num_players = max(players_by_idx.keys()) + 1
            if any(i not in players_by_idx for i in range(num_players)):
                continue
            players = [players_by_idx[i] for i in range(num_players)]

            raw = m.group("input").strip().lower()
            if raw in ("t", "take"):
                label = 1
            elif raw in ("", "p", "pass"):
                label = 0
            else:
                continue

            feats = extract_features(
                players, active_card, tokens_on_card, deck_left, you_idx
            )
            samples.append(
                Sample(
                    features=feats,
                    label=label,
                    you_won=you_won,
                    log_name=path.name,
                )
            )

            game = make_game_state(
                players, active_card, tokens_on_card, deck_left, you_idx
            )
            h1 = heuristic.choose(game, you_idx)
            h2 = heuristic2.choose(game, you_idx)
            baseline["heuristic"]["total"] += 1
            baseline["heuristic2"]["total"] += 1
            if (h1 == "take") == (label == 1):
                baseline["heuristic"]["match"] += 1
            if (h2 == "take") == (label == 1):
                baseline["heuristic2"]["match"] += 1

    return samples, baseline


def standardize(X: List[List[float]]) -> Tuple[List[float], List[float], List[List[float]]]:
    n = len(X)
    dim = len(X[0])
    mean = [0.0] * dim
    for row in X:
        for i, v in enumerate(row):
            mean[i] += v
    mean = [v / n for v in mean]

    std = [0.0] * dim
    for row in X:
        for i, v in enumerate(row):
            std[i] += (v - mean[i]) ** 2
    std = [math.sqrt(v / n) for v in std]

    Xs = []
    for row in X:
        Xs.append(
            [(v - mean[i]) / std[i] if std[i] != 0 else (v - mean[i]) for i, v in enumerate(row)]
        )
    return mean, std, Xs


def train_logistic(
    X: List[List[float]],
    y: List[int],
    weights: Optional[List[float]] = None,
    lr: float = 0.1,
    l2: float = 0.1,
    epochs: int = 800,
) -> Tuple[List[float], float, List[float], List[float]]:
    if weights is None:
        weights = [1.0] * len(y)

    mean, std, Xs = standardize(X)
    dim = len(X[0])
    w = [0.0] * dim
    b = 0.0

    total_w = sum(weights)

    for _ in range(epochs):
        grad_w = [0.0] * dim
        grad_b = 0.0
        for row, yi, wi in zip(Xs, y, weights):
            z = b + sum(wj * xj for wj, xj in zip(w, row))
            if z >= 0:
                p = 1.0 / (1.0 + math.exp(-z))
            else:
                exp_z = math.exp(z)
                p = exp_z / (1.0 + exp_z)
            err = (p - yi) * wi
            grad_b += err
            for i, xj in enumerate(row):
                grad_w[i] += err * xj
        grad_w = [(gw / total_w) + l2 * wj for gw, wj in zip(grad_w, w)]
        grad_b = grad_b / total_w

        w = [wj - lr * gw for wj, gw in zip(w, grad_w)]
        b -= lr * grad_b

    return w, b, mean, std


def predict_proba(
    X: List[List[float]], w: List[float], b: float, mean: List[float], std: List[float]
) -> List[float]:
    probs = []
    for row in X:
        z = b
        for wj, xj, m, s in zip(w, row, mean, std):
            if s == 0:
                z += wj * (xj - m)
            else:
                z += wj * ((xj - m) / s)
        if z >= 0:
            p = 1.0 / (1.0 + math.exp(-z))
        else:
            exp_z = math.exp(z)
            p = exp_z / (1.0 + exp_z)
        probs.append(p)
    return probs


def save_model(
    out_path: Path,
    weights: List[float],
    bias: float,
    mean: List[float],
    std: List[float],
    meta: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "weights": weights,
        "bias": bias,
        "mean": mean,
        "std": std,
        "meta": meta,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train logistic regression from play logs")
    parser.add_argument("--logs", default="logs/play-*.txt", help="Glob for play logs")
    parser.add_argument("--out", required=True, help="Output model path (json)")
    parser.add_argument("--winner-only", action="store_true", help="Use only logs where YOU won")
    parser.add_argument(
        "--invert-losses",
        action="store_true",
        help="Use winner samples + inverted labels from losing logs",
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument(
        "--no-class-balance",
        action="store_false",
        dest="class_balance",
        default=True,
        help="Disable class balancing",
    )
    args = parser.parse_args()

    log_paths = [Path(p) for p in glob.glob(args.logs)]
    if not log_paths:
        raise SystemExit(f"No logs found for pattern: {args.logs}")

    samples: List[Sample] = []
    baseline_totals = {
        "heuristic": {"match": 0, "total": 0},
        "heuristic2": {"match": 0, "total": 0},
    }
    for path in sorted(log_paths):
        s, base = parse_log(path)
        samples.extend(s)
        for key in baseline_totals:
            baseline_totals[key]["match"] += base[key]["match"]
            baseline_totals[key]["total"] += base[key]["total"]

    if args.winner_only and args.invert_losses:
        raise SystemExit("--winner-only and --invert-losses are mutually exclusive.")

    if args.winner_only:
        samples = [s for s in samples if s.you_won]
    elif args.invert_losses:
        winners = [s for s in samples if s.you_won]
        losers_inverted = [
            Sample(
                features=s.features,
                label=1 - s.label,
                you_won=s.you_won,
                log_name=s.log_name,
            )
            for s in samples
            if not s.you_won
        ]
        samples = winners + losers_inverted

    if not samples:
        raise SystemExit("No samples collected.")

    X = [s.features for s in samples]
    y = [s.label for s in samples]

    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        raise SystemExit("Need both TAKE and PASS samples to train.")

    if args.class_balance:
        pos_w = len(y) / (2 * pos)
        neg_w = len(y) / (2 * neg)
        weights = [pos_w if yi == 1 else neg_w for yi in y]
    else:
        weights = [1.0] * len(y)

    w, b, mean, std = train_logistic(
        X, y, weights=weights, lr=args.lr, l2=args.l2, epochs=args.epochs
    )
    probs = predict_proba(X, w, b, mean, std)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    acc = sum(int(p == yi) for p, yi in zip(preds, y)) / len(y)

    meta = {
        "logs": [p.name for p in sorted(log_paths)],
        "winner_only": args.winner_only,
        "invert_losses": args.invert_losses,
        "samples": len(y),
        "take_rate": pos / len(y),
        "train_accuracy": acc,
    }
    save_model(Path(args.out), w, b, mean, std, meta)

    print(f"logs={len(log_paths)} samples={len(y)} take_rate={pos/len(y):.2%}")
    for key in baseline_totals:
        total = baseline_totals[key]["total"]
        if total:
            match = baseline_totals[key]["match"]
            print(f"{key}_match={match/total:.2%} ({match}/{total})")
    print(f"train_accuracy={acc:.2%}")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
