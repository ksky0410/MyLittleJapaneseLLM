# 実験ノート：UnigramとBPEの日本語Tokenizer比較

## 基本情報

- 実験番号：002
- 計画作成日：2026-09-05
- 担当者：ユーザーとCodex
- Gitコミット：689a138。比較スクリプトと実験ノートは、このコミット後の未コミット作業ツリーで実施した
- ブランチ：main
- 状態：完了

計画作成後、2026-09-05に内蔵サンプルから作成した同一のtrain splitで比較を実施した。結果JSONは `experiments/results/2026-09-05_002-tokenizer-comparison.json` に保存した。

## 今回確かめたいこと

同じtrain split、同じ要求語彙数、同じ文字データから学習したSentencePiece UnigramとBPEについて、日本語テキストのToken数、平均文字/token、固定サンプルのpiece分割がどのように違うかを比較する。

## 仮説

日本語の短いdebugコーパスでは、両方式の実際の語彙数が要求値と一致しない可能性がある。UnigramとBPEでは、`むかしむかし`、`人工知能とは`、`今日は良い天気です。` のまとまり方が異なり、その違いが同じ入力のToken数と平均文字/tokenに表れると予想する。ただし、コーパスが小さいため、一般的な日本語Tokenizerの優劣までは判断しない。

## 使用条件

- データ：`artifacts/corpus/train.txt`。実験002ではUnigramとBPEの両方に同じファイルを使う
- データ分割：実験001と同じseed 42、validation ratio 0.2で作成したtrain splitを使う予定
- 要求語彙数：128
- Unigramモデル：`artifacts/tokenizer/unigram.model` を予定
- BPEモデル：`artifacts/tokenizer/bpe.model` を予定
- レポート入力：`artifacts/corpus/train.txt` を予定
- 比較項目：model path、実語彙数、総token数、平均文字/token、固定3サンプルのpiece列

小コーパスで要求語彙数が必要な文字種類数を下回る場合、Tokenizer実装はSentencePieceへ渡す値を自動調整する。その実効値と実際の語彙数は、実験実施後に親側が記録する。

## 実行予定のコマンド

```bash
.venv/bin/python scripts/prepare_data.py \
  --input data/sample_ja.txt \
  --output-dir artifacts/corpus \
  --val-ratio 0.2 \
  --seed 42

.venv/bin/python scripts/train_tokenizer.py \
  --input artifacts/corpus/train.txt \
  --model-prefix artifacts/tokenizer/unigram \
  --vocab-size 128 \
  --model-type unigram

.venv/bin/python scripts/train_tokenizer.py \
  --input artifacts/corpus/train.txt \
  --model-prefix artifacts/tokenizer/bpe \
  --vocab-size 128 \
  --model-type bpe

.venv/bin/python scripts/tokenizer_report.py \
  --tokenizer artifacts/tokenizer/unigram.model \
  --tokenizer artifacts/tokenizer/bpe.model \
  --input artifacts/corpus/train.txt
```

レポートは標準出力のJSONとして取得し、比較結果を `experiments/results/2026-09-05_002-tokenizer-comparison.json` に保存した。比較実験で使った各ファイルのハッシュが必要になる規模では、次回からハッシュも記録する。

## 成功基準

同じ入力に対して2つのTokenizerのJSON要素が出力され、それぞれにmodel path、実語彙数、総token数、平均文字/token、固定3サンプルのpiece列が含まれていることを成功条件とする。Tokenizerの学習に失敗した場合は、要求語彙数、実効語彙数、SentencePieceのエラーを削除せず記録する。数値差が小さい場合も、差が小さいという結果として扱い、優劣を無理に結論づけない。

## 結果

要求語彙数128は、学習データの文字種類を収めるため、UnigramとBPEの両方で実効語彙数299へ自動調整された。両方式とも実際の語彙数は299である。

| Tokenizer | 総Token数 | 平均文字/token |
|---|---:|---:|
| Unigram | 1,341 | 0.9866 |
| BPE | 1,283 | 1.0312 |

BPEはUnigramより58 token少なく、同じ入力に対して約4.3%短いToken列になった。固定サンプル3種類のpiece分割は、この小さなコーパスでは両方式で同じだった。そのため、今回の結果から日本語Tokenizerとしての優劣は判断しない。Token数の差がモデル学習速度やvalidation lossに与える影響は、同じモデルで別々に学習する次の実験で確認する。

## 次に試すこと

同じモデル構成でUnigramとBPEのToken列を使った短い学習をそれぞれ実施し、Token数の差がvalidation lossと学習時間に与える影響を比較する。その後、必要であれば語彙数を一つだけ変更して同じ比較を繰り返す。
