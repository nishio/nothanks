# Heuristic AIのアルゴリズム解説

このドキュメントは `nothanks_cui.py` の `HeuristicAI` が行っている判断ロジックを、日本語で詳細に説明するものです。

## 前提: スコアとΔscore

- **カード点 (`cardPts`)**: 手札の連続ラン（例: 6-9）は最小値だけを足す。
- **総スコア (`score`)**: `score = cardPts - tokens`（低いほど良い）。
- **即時Δscore**（そのカードを今TAKEしたときの増分）:
  - `ΔcardPts = score_cards(cards ∪ {card}) - score_cards(cards)`
  - `Δscore = ΔcardPts - tokens_on_card`

`tokens_on_card` は「カードに載っているチップ数」です。  
TAKEするとチップが手元に来るので、`Δscore` はその分だけ小さくなります。

## アルゴリズム概要（優先順位）

1. **トークンが0なら強制TAKE**
2. **首位ブロック（blocking）**
3. **milking（1回だけPASSしてチップを乗せる）**
4. **メイン閾値ルール**

以下で順に詳しく説明します。

## 1. トークン0なら強制TAKE

PASSできないため必ずTAKEします。

## 2. 首位ブロック（blocking）

近いスコアのトップ争いのとき、  
「相手が取るより自分が取った方が相対的に得」ならTAKEします。

判定条件（コード内のパラメータ名付き）:

- 近さ判定:  
  `my_score - leader_score <= block_range`
- 相対スイング判定:  
  `my_take_cost + leader_take_cost < -block_epsilon`

ここで:

- `my_take_cost` は自分の `Δscore`  
- `leader_take_cost` は「首位プレイヤーがそのカードをTAKEしたときのΔscore」
- `leader` は「自分以外の中で現在スコアが最小の相手」（同点は先に見つかった方）

直感的には、  
**相手にとって得なカードを取らせるより、自分が取って差を詰める**動きです。

## 3. milking（1回だけPASSしてチップを乗せる）

「自分にとって良いカード」を、  
**相手が取りにくいと判断したときだけ1回PASS**してチップを積みます。

判定条件:

- 自分にとって十分良い:  
  `my_take_cost <= milk_good_cost`
- まだチップが少ない:  
  `tokens_on_card < milk_max_tokens_on_card`
- 相手がすぐ取りたくならない:
  - 相手のトークンが少なすぎない: `opp.tokens >= milk_min_opp_tokens`
  - 相手の即時Δscoreが良くない: `opp_take_cost > milk_opp_want_threshold`

この条件をすべて満たすときだけ **PASS** します。  
これにより「少し待ってからおいしく取る」動きになります。

## 4. メイン閾値ルール

最後に、シンプルな閾値判定を行います。

```
th = base_threshold
   + lowtoken_factor / (tokens + 1)
   + endgame_factor * frac_end

frac_end = 1 - remaining_draw / 23
```

- トークンが少ないほど `th` は大きくなり、**TAKEしやすく**なります。
- 終盤になるほど `frac_end` が増え、**TAKE寄り**になります。

最終判定:

- `my_take_cost <= th` なら **TAKE**
- それ以外は **PASS**

## パラメータ一覧（デフォルト）

`HeuristicParams` の初期値は以下です。

```
base_threshold = 0.0
lowtoken_factor = 3.0
endgame_factor = 0.5

block_range = 12.0
block_epsilon = 0.0

milk_good_cost = 0.0
milk_max_tokens_on_card = 2
milk_opp_want_threshold = 0.0
milk_min_opp_tokens = 1
```

## 擬似コード（全体）

```
if tokens == 0:
    TAKE

compute my_take_cost, my_score
compute leader_score, leader_take_cost

if my_score - leader_score <= block_range:
    if my_take_cost + leader_take_cost < -block_epsilon:
        TAKE

if my_take_cost <= milk_good_cost and tokens_on_card < milk_max_tokens_on_card:
    if all(opponent is "safe"):
        PASS

th = base_threshold + lowtoken_factor/(tokens+1) + endgame_factor*frac_end
if my_take_cost <= th:
    TAKE
else:
    PASS
```

## 注意点

- **相手の行動予測は簡易**で、即時Δscoreだけを見ています。  
  深い読み（数手先の探索）はしていません。
- `blocking` と `milking` は **閾値ルールより優先**されます。
