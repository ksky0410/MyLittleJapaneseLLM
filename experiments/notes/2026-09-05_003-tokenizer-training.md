# 実験ノート：UnigramとBPEのdebug学習比較

## 基本情報

- 実験番号：003
- 計画作成日：2026-09-05
- 担当者：ユーザーとCodex
- Gitコミット：53c96ed。設定ファイルと実験ノートは、このコミット後の未コミット作業ツリーで実施した
- ブランチ：main
- 状態：完了

計画作成後、2026-09-05にUnigramとBPEのdebug学習、評価、生成を実施した。結果JSONは `experiments/results/2026-09-05_003-tokenizer-training.json` に保存した。

## 今回確かめたいこと

同一の日本語train/validation splitから学習したUnigram TokenizerとBPE Tokenizerを使い、同じdecoder-only Transformerを同じ条件で100 step学習する。Tokenizer方式の違いが、validation loss、validation perplexity、学習時間、固定promptの生成結果にどのように現れるかを比較する。

## 仮説

UnigramとBPEではpieceの分割が異なるため、同じ文字量でも入力Token数と学習で処理するTokenの内容が変わる。小さな日本語コーパスでは、どちらか一方が明確に優れるとは限らず、実際の語彙数やToken分割の違いがvalidation lossと生成の繰り返し方に影響すると予想する。データ量が少ないため、今回の結果だけでTokenizerの一般的な優劣は判断しない。

## 比較条件

- データ：`artifacts/corpus/train.txt` と `artifacts/corpus/val.txt`
- Tokenizer：同じtrain splitから作ったSentencePiece UnigramとBPE
- 要求語彙数：128
- モデル：dim 64、2層、4 heads、context length 64、MLP倍率4、入力Embeddingと出力weightを共有
- 学習：AdamW、batch size 8、最大100 step、seed 42
- 学習率：3e-4、10 step warmup後のcosine decay、最小学習率3e-5
- 評価間隔：20 step、validation batch数4
- 生成：prompt `むかしむかし、`、最大80 token、temperature 0.8、top-k 20
- 分離：Tokenizer、Token列、checkpoint、sampleをUnigram/BPEごとに別ディレクトリへ保存した

使用する設定ファイルは、条件を揃えた`configs/debug-unigram.toml`と`configs/debug-bpe.toml`である。

## 実行予定のコマンド

```bash
.venv/bin/python scripts/prepare_data.py --input data/sample_ja.txt --output-dir artifacts/corpus --val-ratio 0.2 --seed 42

.venv/bin/python scripts/train_tokenizer.py --input artifacts/corpus/train.txt --model-prefix artifacts/tokenizer/unigram --vocab-size 128 --model-type unigram
.venv/bin/python scripts/train_tokenizer.py --input artifacts/corpus/train.txt --model-prefix artifacts/tokenizer/bpe --vocab-size 128 --model-type bpe

.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/unigram.model --input artifacts/corpus/train.txt --output artifacts/tokens/unigram-train.bin
.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/unigram.model --input artifacts/corpus/val.txt --output artifacts/tokens/unigram-val.bin
.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/bpe.model --input artifacts/corpus/train.txt --output artifacts/tokens/bpe-train.bin
.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/bpe.model --input artifacts/corpus/val.txt --output artifacts/tokens/bpe-val.bin

.venv/bin/python scripts/train.py --config configs/debug-unigram.toml
.venv/bin/python scripts/train.py --config configs/debug-bpe.toml
.venv/bin/python scripts/evaluate.py --config configs/debug-unigram.toml --checkpoint artifacts/checkpoints/unigram/step_000100.npz
.venv/bin/python scripts/evaluate.py --config configs/debug-bpe.toml --checkpoint artifacts/checkpoints/bpe/step_000100.npz
```

Tokenizerのpiece分割自体は、実験002で追加した次のコマンドで比較する。

```bash
.venv/bin/python scripts/tokenizer_report.py \
  --tokenizer artifacts/tokenizer/unigram.model \
  --tokenizer artifacts/tokenizer/bpe.model \
  --input artifacts/corpus/train.txt
```

## 成功基準

両方式について学習が最大100 stepまで完走し、各設定専用のmetrics JSONL、validation loss、perplexity、checkpoint metadata、固定promptのsampleが保存されることを成功条件とする。比較表には、実際の学習時間、最終および最良validation loss、perplexity、生成結果の保存先を記録する。途中停止やエラーが起きた場合は、成功扱いにせず、発生した方式と原因をそのまま記録する。

## 結果

Unigramは1,387 token、BPEは1,329 token（いずれも文末EOSを含む）の学習列になった。BPEは処理Token数が約4.2%少ない。

| Tokenizer | 最終train loss | 最終validation loss | validation perplexity | 学習時間（秒） |
|---|---:|---:|---:|---:|
| Unigram | 2.7415 | 3.9557 | 52.23 | 1.76 |
| BPE | 2.8895 | 4.1653 | 64.41 | 0.62 |

同じモデル、同じseed、同じ100 stepという条件では、Unigramのvalidation lossがBPEより0.2096低く、perplexityも52.23対64.41で低かった。ただし、これは1,323文字程度の非常に小さなコーパスでの結果であり、Tokenizer方式そのものの一般的な優劣とは解釈しない。

固定prompt `むかしむかし、` からの生成は、Unigramが「むかしむかし、小さない作が咲い作が駅になできた。」、BPEが「むかしむかし、好実が動こと、使がしました。」となった。どちらも日本語らしい断片はあるものの、文法と意味の連続性は不十分である。

両方式のcheckpoint、metrics JSONL、生成サンプルは、それぞれ `artifacts/checkpoints/unigram/`、`artifacts/checkpoints/bpe/`、`artifacts/samples/unigram/`、`artifacts/samples/bpe/` に保存した。

## 次に試すこと

次はモデル条件とTokenizer方式を固定し、学習データ量を増やす。今回の差が小規模コーパス特有のものかを確認するため、データ量だけを変更してvalidation lossと生成を比較する。
