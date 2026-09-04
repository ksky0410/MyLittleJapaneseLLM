# 実験ノート：RoPEを導入した混合日本語5Mモデル

## 基本情報

- 実験番号：010
- 記録日：2026-09-05
- 担当者：ユーザーとCodex
- 実験開始時のGitコミット：`4691dae`
- ブランチ：`main`
- 状態：学習開始前

## 今回確かめたいこと

実験009の混合日本語5Mモデルを基準に、位置表現だけを学習可能なabsolute position embeddingからRoPEへ変更した場合の差を確認する。RoPEはqとkへ適用し、baseは10,000とする。LayerNorm、通常のmulti-head attention、モデルの層数・次元数・語彙・optimizer・データ・seedは変更しない。

仮説は、context 256の短いsmokeではvalidation lossの差は小さいかもしれないが、RoPEの相対的な位置情報によって長い文脈へ拡張しやすい性質が現れる可能性があるというものである。一方、今回のデータ量とモデルサイズでは、位置表現の差よりもデータ源の偏りやTokenizerの影響が大きい可能性もある。結果が悪くても、RoPEが常に優れる・劣るとは解釈しない。

## 実験条件

基準と比較対象は同じ混合train Token列`artifacts/tokens/mixed-ja-80-10-10-v1-train.bin`と、一般validation Token列`artifacts/tokens/aozora-neko-formal-v2-val.bin`である。Tokenizerは実験009のSentencePiece Unigram 4,096語彙をそのまま使う。学習設定は`configs/rope-mixed-ja-5m-smoke.toml`で、dim 240、6層、6 heads、context 256、MLP ratio 4、batch size 8、AdamW、learning rate 3e-4から3e-5、warmup 300、weight decay 0.1、seed 42、500 stepである。

absolute基準の概算パラメータ数は5,197,920、RoPEは位置embeddingを持たないため5,136,480である。checkpointとsampleは基準とは別の`artifacts/checkpoints/rope-mixed-ja-5m-smoke/`と`artifacts/samples/rope-mixed-ja-5m-smoke/`へ保存する。RoPE checkpointのmetadataには`position_embedding: rope`を保存し、absolute設定で誤って読み込めないことも確認する。

実行コマンドは次のとおりである。

```bash
.venv/bin/python scripts/train.py \
  --config configs/rope-mixed-ja-5m-smoke.toml
```

学習後には、step 0、100、200、300、400、500の生成文を保存し、step 500 checkpointをreloadして一般・会話・医療のdomain別評価を行う。基準実験009と同じprompt、eval_batches、評価Token列を使う。

## 成功条件

500 stepまでNaN、OOM、途中停止なく完走し、RoPEのforward shape、checkpoint保存、reload、domain別評価が通ること。比較では、最良validation loss・perplexity・所要時間・生成文の崩れ方を記録する。RoPEが悪化した場合も削除せず、その理由を仮説として残す。

## 実行結果

RoPE学習は500 stepまでエラー、NaN、OOM、途中停止なく完走し、所要時間は79.53秒であった。stepごとの主要値は次のとおりである。

| step | train loss | validation loss | validation perplexity |
|---:|---:|---:|---:|
| 1 | 8.9089 | 8.8342 | 6,865.24 |
| 100 | 5.5780 | 6.7096 | 820.26 |
| 200 | 5.1642 | 6.3333 | 563.03 |
| 300 | 5.7729 | 5.9549 | 385.63 |
| 400 | 3.7901 | 5.6826 | 293.70 |
| 500 | 4.0454 | 5.5442 | 255.76 |

最良checkpointは`artifacts/checkpoints/rope-mixed-ja-5m-smoke/step_000500.npz`で、validation loss 5.5442326864、perplexity 255.7582561343である。実験009のabsolute基準に対してvalidation lossは0.2027、perplexityは57.46下がった。RoPEのdomain別評価は`artifacts/evaluations/rope-mixed-ja-5m-smoke-domains.json`に保存した。

| domain | token数 | validation loss | perplexity |
|---|---:|---:|---:|
| 一般 | 11,367 | 5.5442 | 255.76 |
| 会話 | 1,163,469 | 3.3523 | 28.57 |
| 医療 | 199,138 | 4.5273 | 92.51 |

absolute基準との差は、一般-0.2027、会話-0.2009、医療-0.3814で、今回の短いrunではすべてのdomainでRoPE側が低かった。ただし、同じseedでもabsolute側は学習可能なposition embeddingの乱数初期化を一つ余分に行うため、qkvなどの初期値まで完全に同じではない。したがって、この結果はRoPEの優位性を確定するものではなく、同じ初期値を厳密にそろえた追試が必要である。また、最初のMLX実行に伴うコンパイル時間の差もあり、所要時間の比較は参考値にとどめる。

reload後の生成文は、一般promptでは`今日はは`、物語promptでは短い文、会話promptでは`こんにちは。`、医療promptでは問題番号や選択肢らしい形式を含む出力となった。生成文は[reloaded-today.txt](../../artifacts/samples/rope-mixed-ja-5m-smoke/reloaded-today.txt)、[reloaded-story.txt](../../artifacts/samples/rope-mixed-ja-5m-smoke/reloaded-story.txt)、[reloaded-conversation.txt](../../artifacts/samples/rope-mixed-ja-5m-smoke/reloaded-conversation.txt)、[reloaded-medical.txt](../../artifacts/samples/rope-mixed-ja-5m-smoke/reloaded-medical.txt)から確認できる。会話のmarkerは生成時のTokenizer正規化で空白区切りになっており、会話継続の品質は別評価が必要である。

metrics、summary、checkpoint metadata、生成文、domain別評価JSONはGitHubへcommit・pushする。巨大な`.npz`とToken・Tokenizer本体はGit管理対象外とし、manifestと実験ノートへ条件とSHA-256を残す。次は、一般sourceを増やしてtoken比率を記録する実験か、初期値を厳密にそろえたRoPE追試のどちらか一つを行う。
