# nothanks_cui

1ファイル完結の Python スクリプトで、No Thanks! のCUI対戦と自己対戦シミュレーションを行えます。

## できること

- 人間1 + AI2でCUI対戦
- AI3体の自己対戦を大量に回して統計（平均スコア/勝率など）を取得
- ルールベースAI（greedy / heuristic）を標準搭載

## 使い方

### 人間1 + AI2（CUI対戦）

```bash
python nothanks_cui.py play
```

AIを指定したい場合（例：heuristic と greedy）：

```bash
python nothanks_cui.py play --ai heuristic greedy
```

自分の席（0/1/2）を変える：

```bash
python nothanks_cui.py play --seat 1 --ai heuristic greedy
```

人間ターンで表示される計算ヒントを消す：

```bash
python nothanks_cui.py play --no-math-hint
```

（デバッグ用）除外9枚を表示：

```bash
python nothanks_cui.py play --show-removed
```

### AI3体の自己対戦（統計）

```bash
python nothanks_cui.py simulate --games 10000 --seed 1 --ai heuristic greedy greedy
```

ログを少し見たい（最初の数ゲームだけ詳細）：

```bash
python nothanks_cui.py simulate --games 50 --verbose
```

出力は次のような統計を表示します。

- 各AIの平均スコア（低いほど良い）
- 標準偏差
- 勝率（同点は割り勘）

## 実装してあるAI

- `random`：ランダム（比較用）
- `greedy`：即時のΔscore <= 0ならTake、それ以外はPass
- `heuristic`：ルールベース強め（パラメータ入り）
  - Takeの即時価値：`Δscore = Δ(card_points) - tokens_on_card`
  - トークンが少ないほどPassのオプション価値を重く見てTake寄りにする
  - 終盤は少しTake寄り
  - トップ争い（近い点差）では相対スイングを見てTakeすることがある
  - 自分に良いカードを安全っぽいなら1回だけ回してチップを乗せる（milking）ことがある
- `lr1`：人間ログから学習したロジスティック回帰（確率的に行動）
- `lr2`：勝ちログのみで学習したロジスティック回帰（確率的に行動）

## パラメータ調整

スクリプト内の `HeuristicParams` を編集するとすぐ反映されます。

## 学習モデルの作成

ログから学習モデルを作るには `scripts/train_lr.py` を使います。

```bash
python scripts/train_lr.py --out models/lr1.json
python scripts/train_lr.py --out models/lr2.json --winner-only
```

## ドキュメント

- Heuristic AIの詳細: `docs/heuristic.md`

## 参考（今後の拡張案）

- ルールベースのパラメータを自己対戦でスイープして最適化
- 軽い探索（1手〜数手のロールアウト）を追加
- 自己対戦ログから弱点診断レポートを作成
