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

学習と評価の直後に、metrics、summary、checkpoint metadata、生成文へのリンク、domain別評価JSONを追記する。巨大な`.npz`とToken・Tokenizer本体はGit管理対象外とし、軽量な生成文・metrics・metadata・評価JSON・このノートはGitHubへcommit・pushする。
