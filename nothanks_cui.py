
#!/usr/bin/env python3
"""
No Thanks! (Geschenkt) - CUI + simple rule-based AIs

Modes:
  - play:     1 Human + 2 AIs (CUI)
  - simulate: 3 AIs self-play and collect statistics

Rules follow the English rulebook:
- Cards 3..35 (33 cards), remove 9 unseen, play with remaining 24.
- On your turn: PASS (pay 1 token, card gets +1 token, turn passes left)
               or TAKE (take card + all tokens on it, draw new active card and continue).
- Score = sum(lowest card of each run) - remaining tokens (lower is better).

This file is intentionally compact and hackable.
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Tuple


# ---------------------------
# Scoring helpers
# ---------------------------

def score_cards(cards: Set[int]) -> int:
    """Card points: sum of the lowest card in each consecutive run."""
    if not cards:
        return 0
    s = set(cards)
    total = 0
    for c in s:
        if (c - 1) not in s:
            total += c
    return total

def score_player(cards: Set[int], tokens: int) -> int:
    """Total score (lower is better)."""
    return score_cards(cards) - tokens

def runs_str(cards: Set[int]) -> str:
    """Pretty-print runs like '8 | 13-15 | 17'."""
    if not cards:
        return "(none)"
    arr = sorted(cards)
    runs: List[Tuple[int, int]] = []
    start = prev = arr[0]
    for x in arr[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))
    out = []
    for a, b in runs:
        out.append(str(a) if a == b else f"{a}-{b}")
    return " | ".join(out)

def marginal_card_points(cards: Set[int], new_card: int) -> int:
    """Delta of card-points if new_card is added."""
    before = score_cards(cards)
    after = score_cards(set(cards) | {new_card})
    return after - before


# ---------------------------
# Core game engine
# ---------------------------

@dataclass
class PlayerState:
    name: str
    tokens: int
    cards: Set[int] = field(default_factory=set)

class NoThanksGame:
    """Engine for a single round."""
    def __init__(self, players: List[PlayerState], seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.players = players
        self.n = len(players)

        self.removed: List[int] = []
        self.deck: List[int] = []
        self.active_card: Optional[int] = None
        self.tokens_on_card: int = 0
        self.current: int = 0  # index of player whose turn it is

        self.log: List[str] = []

    @staticmethod
    def starting_tokens(num_players: int) -> int:
        if 3 <= num_players <= 5:
            return 11
        if num_players == 6:
            return 9
        if num_players == 7:
            return 7
        raise ValueError("No Thanks supports 3–7 players")

    def setup(self, start_player: Optional[int] = None) -> None:
        cards = list(range(3, 36))
        self.rng.shuffle(cards)
        self.removed = cards[:9]
        self.deck = cards[9:]  # 24 cards

        self.active_card = self.deck.pop(0)
        self.tokens_on_card = 0

        self.current = self.rng.randrange(self.n) if start_player is None else (start_player % self.n)
        self.log.append(f"Setup: active={self.active_card}, start={self.players[self.current].name}")

    def remaining_draw(self) -> int:
        return len(self.deck)

    def is_over(self) -> bool:
        return self.active_card is None

    def apply_pass(self) -> None:
        ps = self.players[self.current]
        if ps.tokens <= 0:
            raise RuntimeError("Cannot PASS with 0 tokens.")
        ps.tokens -= 1
        self.tokens_on_card += 1
        self.log.append(f"{ps.name}: PASS (pays 1) -> tokens_on_card={self.tokens_on_card}")
        self.current = (self.current + 1) % self.n

    def apply_take(self) -> None:
        ps = self.players[self.current]
        card = self.active_card
        assert card is not None

        ps.cards.add(card)
        ps.tokens += self.tokens_on_card
        self.log.append(f"{ps.name}: TAKE {card} (+{self.tokens_on_card} tokens)")
        self.tokens_on_card = 0

        if self.deck:
            self.active_card = self.deck.pop(0)
            self.log.append(f"New active={self.active_card} (same player continues)")
        else:
            self.active_card = None
            self.log.append("Deck empty -> round ends")

    def provisional_scores(self) -> List[int]:
        return [score_player(p.cards, p.tokens) for p in self.players]


# ---------------------------
# Agents (Human / AI)
# ---------------------------

class Agent:
    def __init__(self, name: str):
        self.name = name

    def choose(self, game: NoThanksGame, idx: int) -> str:
        """Return 'take' or 'pass'."""
        raise NotImplementedError

class HumanCUI(Agent):
    def __init__(self, name: str, show_math: bool = True):
        super().__init__(name)
        self.show_math = show_math

    def choose(self, game: NoThanksGame, idx: int) -> str:
        ps = game.players[idx]
        if ps.tokens == 0:
            return "take"

        while True:
            if self.show_math:
                card = game.active_card
                assert card is not None
                take_delta = marginal_card_points(ps.cards, card)
                take_cost = take_delta - game.tokens_on_card
                print(f"(info) If TAKE now: Δscore = {take_cost:+d} (ΔcardPts={take_delta:+d}, tokens_on_card={game.tokens_on_card}).")
                print("(info) If PASS now: Δscore = +1 (spend 1 token).")
                t_on_pass = game.tokens_on_card + 1
                for j, opp in enumerate(game.players):
                    if j == idx:
                        continue
                    opp_take_cost = marginal_card_points(opp.cards, card) - t_on_pass
                    print(f"(info) If PASS and {opp.name} TAKES: {opp.name} Δscore = {opp_take_cost:+d}.")

            cmd = input("Your move: [t]ake / [p]ass (default: pass) / [?]help > ").strip().lower()
            if cmd in ("", "p", "pass"):
                return "pass"
            if cmd in ("t", "take"):
                return "take"
            if cmd in ("?", "h", "help"):
                print("Rules reminder:")
                print("- PASS: pay 1 token onto the card; turn passes left.")
                print("- TAKE: take the card + all tokens on it; draw new card and continue your turn.")
                continue

class RandomAI(Agent):
    def choose(self, game: NoThanksGame, idx: int) -> str:
        ps = game.players[idx]
        if ps.tokens == 0:
            return "take"
        return "take" if game.rng.random() < 0.5 else "pass"

class GreedyAI(Agent):
    """Myopic baseline: take if immediate Δscore <= 0, else pass if possible."""
    def choose(self, game: NoThanksGame, idx: int) -> str:
        ps = game.players[idx]
        if ps.tokens == 0:
            return "take"
        card = game.active_card
        assert card is not None
        take_cost = marginal_card_points(ps.cards, card) - game.tokens_on_card
        return "take" if take_cost <= 0 else "pass"

@dataclass
class HeuristicParams:
    # Main threshold: take if (Δscore <= threshold)
    base_threshold: float = 0.0
    lowtoken_factor: float = 3.0
    endgame_factor: float = 0.5

    # Blocking (near zero-sum among top contenders)
    block_range: float = 12.0
    block_epsilon: float = 0.0

    # Milking (passing once on a card you like to accumulate tokens)
    milk_good_cost: float = 0.0
    milk_max_tokens_on_card: int = 2
    milk_opp_want_threshold: float = 0.0
    milk_min_opp_tokens: int = 1

class HeuristicAI(Agent):
    """
    A compact rule-based AI.

    Key ideas:
    - Evaluate TAKE by its immediate Δscore = Δ(card_points) - tokens_on_card.
    - Allow taking slightly "bad" cards when low on tokens (option value).
    - Sometimes TAKE to block the current leader if the swing is large.
    - Sometimes PASS on a good card to "milk" extra tokens, if likely safe.
    """
    def __init__(self, name: str, params: Optional[HeuristicParams] = None):
        super().__init__(name)
        self.p = params or HeuristicParams()

    def _take_threshold(self, tokens: int, remaining_draw: int) -> float:
        # remaining_draw: 0..23
        frac_end = 1.0 - (remaining_draw / 23.0 if 23.0 else 1.0)
        return (
            self.p.base_threshold
            + self.p.lowtoken_factor / (tokens + 1.0)
            + self.p.endgame_factor * frac_end
        )

    def choose(self, game: NoThanksGame, idx: int) -> str:
        me = game.players[idx]
        if me.tokens == 0:
            return "take"

        card = game.active_card
        assert card is not None
        t_on = game.tokens_on_card
        remaining = game.remaining_draw()

        my_take_cost = marginal_card_points(me.cards, card) - t_on
        my_score = score_player(me.cards, me.tokens)

        opp_indices = [j for j in range(game.n) if j != idx]
        opp_scores = {j: score_player(game.players[j].cards, game.players[j].tokens) for j in opp_indices}
        leader = min(opp_scores, key=lambda j: opp_scores[j])
        leader_score = opp_scores[leader]

        # 1) Block leader if the relative swing is big and competition is close.
        if my_score - leader_score <= self.p.block_range:
            leader_ps = game.players[leader]
            leader_take_cost = marginal_card_points(leader_ps.cards, card) - t_on
            # If (me TAKE) is better than (leader TAKE), in a 2-player-relative sense:
            # prefer TAKE when my_take_cost + leader_take_cost < 0.
            if my_take_cost + leader_take_cost < -self.p.block_epsilon:
                return "take"

        # 2) Milk: pass once on a card that is already good for us and seems bad for others.
        if my_take_cost <= self.p.milk_good_cost and t_on < self.p.milk_max_tokens_on_card:
            safe = True
            for j in opp_indices:
                opp = game.players[j]
                if opp.tokens < self.p.milk_min_opp_tokens:
                    safe = False
                    break
                opp_take_cost = marginal_card_points(opp.cards, card) - t_on
                if opp_take_cost <= self.p.milk_opp_want_threshold:
                    safe = False
                    break
            if safe:
                return "pass"

        # 3) Main threshold rule.
        th = self._take_threshold(me.tokens, remaining)
        return "take" if my_take_cost <= th else "pass"


AI_CHOICES = ("random", "greedy", "heuristic")

def make_ai(ai_type: str, name: str) -> Agent:
    ai_type = ai_type.lower()
    if ai_type == "random":
        return RandomAI(name)
    if ai_type == "greedy":
        return GreedyAI(name)
    if ai_type == "heuristic":
        return HeuristicAI(name)
    raise ValueError(f"Unknown AI type: {ai_type}")


# ---------------------------
# CUI rendering
# ---------------------------

def render(game: NoThanksGame, show_removed: bool = False) -> None:
    print()
    print("=" * 72)
    if show_removed:
        print(f"Removed(9): {sorted(game.removed)}")
    card_str = "-" if game.active_card is None else str(game.active_card)
    print(f"[{card_str}]-{game.tokens_on_card} (Active card: {card_str}   Tokens on card: {game.tokens_on_card}   Deck left: {game.remaining_draw()})")
    print("-" * 72)
    scores = game.provisional_scores()
    for i, p in enumerate(game.players):
        turn = ">" if i == game.current else " "
        print(f"{turn}[{i}] {p.name:10s}  tokens={p.tokens:2d}  cards={runs_str(p.cards):25s}  "
              f"cardPts={score_cards(p.cards):3d}  score={scores[i]:4d}")
    print("=" * 72)

def print_final(game: NoThanksGame) -> None:
    print("\nFINAL RESULTS")
    scores = game.provisional_scores()
    order = sorted(range(game.n), key=lambda i: scores[i])  # low score wins
    for rank, i in enumerate(order, start=1):
        p = game.players[i]
        print(f"{rank:>2}. {p.name:10s} score={scores[i]:4d}  cardPts={score_cards(p.cards):3d}  tokens={p.tokens:2d}  cards={runs_str(p.cards)}")
    winners = [i for i in range(game.n) if scores[i] == min(scores)]
    if len(winners) == 1:
        print(f"Winner: {game.players[winners[0]].name}")
    else:
        print("Winners:", ", ".join(game.players[i].name for i in winners))


# ---------------------------
# Modes
# ---------------------------

def play_cui(seed: Optional[int], seat: int, ai_types: Tuple[str, str], show_removed: bool, show_math: bool) -> None:
    n = 3
    tokens = NoThanksGame.starting_tokens(n)

    names = [f"Player{i}" for i in range(n)]
    names[seat] = "YOU"

    players = [PlayerState(names[i], tokens) for i in range(n)]
    game = NoThanksGame(players, seed=seed)
    game.setup()

    agents: List[Agent] = [None] * n  # type: ignore
    agents[seat] = HumanCUI("YOU", show_math=show_math)

    ai_slots = [i for i in range(n) if i != seat]
    agents[ai_slots[0]] = make_ai(ai_types[0], names[ai_slots[0]])
    agents[ai_slots[1]] = make_ai(ai_types[1], names[ai_slots[1]])

    # Main loop
    while not game.is_over():
        render(game, show_removed=show_removed)
        idx = game.current
        agent = agents[idx]

        action = agent.choose(game, idx)
        ps = game.players[idx]
        if ps.tokens == 0:
            action = "take"
            print(f"{ps.name} has 0 tokens -> forced TAKE.")

        if action == "pass":
            game.apply_pass()
        else:
            game.apply_take()

    render(game, show_removed=show_removed)
    print_final(game)

def simulate(games: int, seed: int, ai_types: Tuple[str, str, str], verbose: bool) -> None:
    rng = random.Random(seed)
    n = 3
    tokens = NoThanksGame.starting_tokens(n)

    agents = [make_ai(ai_types[i], f"AI{i}") for i in range(n)]

    all_scores: List[List[int]] = []
    win_counts = [0.0] * n

    for gidx in range(games):
        players = [PlayerState(f"AI{i}", tokens) for i in range(n)]
        game = NoThanksGame(players, seed=rng.randrange(1 << 30))
        game.setup()

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

        scores = game.provisional_scores()
        all_scores.append(scores)

        best = min(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        for i in winners:
            win_counts[i] += 1.0 / len(winners)

        if verbose and gidx < 5:
            print("\n--- Game", gidx, "---")
            for line in game.log[-30:]:
                print(line)
            print("Scores:", scores)

    # Stats
    means = [statistics.mean([s[i] for s in all_scores]) for i in range(n)]
    stdevs = [statistics.pstdev([s[i] for s in all_scores]) for i in range(n)]
    total_wins = sum(win_counts)

    print("\nSIMULATION RESULTS")
    print(f"games={games} seed={seed} AIs={ai_types}")
    print("-" * 60)
    for i in range(n):
        print(f"AI{i} ({ai_types[i]:9s})  mean={means[i]:6.2f}  sd={stdevs[i]:6.2f}  winrate={win_counts[i]/games:6.2%}")
    print("-" * 60)
    # sanity
    print(f"(winrates sum) {total_wins/games:6.2%}  (ties are split)")

def main() -> None:
    parser = argparse.ArgumentParser(description="No Thanks! CUI + simple AIs")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_play = sub.add_parser("play", help="Play 1 Human vs 2 AIs (CUI)")
    p_play.add_argument("--seed", type=int, default=None, help="Random seed")
    p_play.add_argument("--seat", type=int, default=0, help="Your seat index (0/1/2)")
    p_play.add_argument("--ai", nargs=2, default=("heuristic", "greedy"), choices=AI_CHOICES,
                        help="Two AI types for the non-human seats, in seat order.")
    p_play.add_argument("--show-removed", action="store_true", help="Debug: show the 9 removed cards")
    p_play.add_argument("--no-math", action="store_true", help="Hide Δscore hint during human turns")

    p_sim = sub.add_parser("simulate", help="Run 3 AIs and collect stats")
    p_sim.add_argument("--games", type=int, default=10000)
    p_sim.add_argument("--seed", type=int, default=0)
    p_sim.add_argument("--ai", nargs=3, default=("heuristic", "greedy", "greedy"), choices=AI_CHOICES)
    p_sim.add_argument("--verbose", action="store_true", help="Print logs for first few games")

    args = parser.parse_args()

    if args.mode == "play":
        play_cui(
            seed=args.seed,
            seat=args.seat,
            ai_types=(args.ai[0], args.ai[1]),
            show_removed=args.show_removed,
            show_math=(not args.no_math),
        )
    elif args.mode == "simulate":
        simulate(
            games=args.games,
            seed=args.seed,
            ai_types=(args.ai[0], args.ai[1], args.ai[2]),
            verbose=args.verbose,
        )

if __name__ == "__main__":
    main()
