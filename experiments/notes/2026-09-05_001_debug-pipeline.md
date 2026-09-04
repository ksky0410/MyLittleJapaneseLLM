# 実験ノート：日本語debugパイプライン

## 基本情報

- 実験番号：001
- 計画作成日：2026-09-05
- 担当者：ユーザーとCodex
- Gitコミット：未コミットの作業ツリー
- ブランチ：main
- 状態：完了

計画作成後、2026-09-05 03:50頃に内蔵サンプルを使ったdebugパイプラインを実施した。実装検証と短いdebug学習を分けて記録する。

## 今回確かめたいこと

内蔵した小さな日本語コーパスからSentencePiece Tokenizerを学習し、学習・検証を別々のToken列へ変換する。その後、MLXのdecoder-only Transformerがcausal next-token lossを計算し、validation lossを評価し、安全なcheckpointと固定プロンプトの生成結果を保存できるかを確かめる。

## 事前の予想

内蔵サンプルは少量なので、一般的な日本語能力は得られないと予想する。一方、100 step程度のdebug学習が正常に進めば、lossとperplexityを出力でき、学習前後で固定プロンプトの出力が変化することを期待する。データが少ないため、生成の繰り返しやvalidation lossの悪化は失敗とは扱わず、観察結果として記録する。

## 使用するデータ

- データセット名・取得元：リポジトリ内の `data/sample_ja.txt`。外部ネットワークから取得しない
- 使用する範囲：内蔵サンプル全体。空行を除き、1行を1文または1段落として扱う
- 前処理：Unicode NFKC、タブの空白化、前後空白の除去、連続空白の整理、空行の除外
- 分割：seed 42で決定的にshuffleし、validation ratio 0.2で文書単位に分割する
- 学習トークン数：1,387 token
- 検証トークン数：354 token
- 重複除去や除外条件：現時点では重複除去なし。データを増やす段階で重複と定型文を確認する

## Tokenizer

- Tokenizerの種類：SentencePiece Unigram。BPEも同じスクリプトで選択可能
- 語彙数：128を目標値とする。小コーパスで必要な文字数が多い場合は、必要語彙数まで自動的に引き上げる
- 学習データ：`artifacts/corpus/train.txt`
- Tokenizerのファイル：`artifacts/tokenizer/ja.model`
- 実際の語彙数：299。要求値128では小コーパスの文字種類を収められなかったため、SentencePieceへ渡す語彙数を299へ自動調整した
- 確認した分割例：Tokenizer学習とToken化は成功。詳細なpiece一覧の比較は次のTokenizer比較実験で行う

## モデルと学習設定

- パラメータ数：122,176。`scripts/inspect_model.py` の概算値
- 層数：2
- モデル次元：64
- Attention head数：4
- コンテキスト長：64
- MLP倍率：4
- バッチサイズ：8
- オプティマイザ：AdamW
- 学習率：3e-4
- 学習率スケジュール：10 stepのwarmup後にcosine decay、最小学習率3e-5
- 乱数シード：42
- 使用デバイス：Apple Silicon MacのMLX
- 最大step：100
- ログ間隔：20 step以下。debugでは1000 stepを超えて記録を空けない

## 実行予定のコマンド

```bash
.venv/bin/python scripts/prepare_data.py --input data/sample_ja.txt --output-dir artifacts/corpus --val-ratio 0.2 --seed 42
.venv/bin/python scripts/train_tokenizer.py --input artifacts/corpus/train.txt --model-prefix artifacts/tokenizer/ja --vocab-size 128 --model-type unigram
.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/ja.model --input artifacts/corpus/train.txt --output artifacts/tokens/train.bin
.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/ja.model --input artifacts/corpus/val.txt --output artifacts/tokens/val.bin
.venv/bin/python scripts/inspect_model.py --config configs/debug.toml
.venv/bin/python scripts/train.py --config configs/debug.toml
.venv/bin/python scripts/evaluate.py --config configs/debug.toml --checkpoint artifacts/checkpoints/step_000100.npz
.venv/bin/python scripts/generate.py --config configs/debug.toml --checkpoint artifacts/checkpoints/step_000100.npz
```

## 期待する結果と判定基準

Tokenizerモデルと学習・検証Token列が作成され、Token列がcontext lengthより長いことを確認する。モデル形状の確認が成功し、学習がstep 100までエラーなく進み、validation loss・perplexity・checkpoint・固定promptサンプルが保存されれば、debugパイプラインの成功とする。生成文の自然さは、この小さなdebugデータだけでは性能指標にしない。

## 実験中の記録

2026-09-05 03:50頃に、最終版コードで学習を実施した。異常、設定変更、途中停止はなかった。

| step | train loss | validation loss | perplexity | 学習率 | 経過時間（秒） |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.3259 | 6.3212 | 556.25 | 0.0000300 | 0.13 |
| 20 | 4.3580 | 4.5077 | 90.71 | 0.0002934 | 0.22 |
| 40 | 3.5678 | 4.1412 | 62.88 | 0.0002365 | 0.31 |
| 60 | 3.1956 | 4.0080 | 55.04 | 0.0001462 | 0.39 |
| 80 | 2.9212 | 3.9622 | 52.57 | 0.0000647 | 0.50 |
| 100 | 2.7415 | 3.9557 | 52.23 | 0.0000301 | 0.60 |

train lossは6.3259から2.7415へ、validation lossは6.3212から3.9557へ低下した。小さなコーパスを使ったdebugとして、next-token学習と検証計算が進んでいることを確認できた。

## 生成サンプル

生成結果は `artifacts/samples/` に保存した。学習前はランダムな文字列だったが、step 100では次のような日本語らしい断片が現れた。ただし、文法的にはまだ不自然であり、性能向上とは解釈しない。

```text
prompt: むかしむかし、

むかしむかし、包、材きます。
```

学習後に別途生成した出力は次のとおりである。

```text
むかしむかし、小さない作が咲い作が駅になできた。
```

## 最終結果

- 最終step：100
- 最終train loss：2.7415
- 最終validation loss：3.9557
- 最終perplexity：52.23
- 最良checkpoint：`artifacts/checkpoints/step_000100.npz`
- 学習時間：約0.61秒
- 最大メモリ使用量：未計測
- 停止理由：予定どおり完了

## 問題と未解決事項

MLXとSentencePieceはプロジェクトの `.venv` に導入した。pytestは8件すべて成功し、ruff checkとruff format --checkも成功した。MLX forward、optimizer update、checkpoint保存とmetadata検証付きの再ロード、evaluate、generateまで確認済みである。最大メモリ使用量はまだ記録していないため、規模を大きくする実験では計測する。

## 次に試すこと

次の比較実験では、語彙数またはcontext lengthのどちらか一つだけを変える。まずは同じ内蔵サンプルでTokenizerのpiece分割とvalidation lossへの影響を比較し、その後にデータ量を増やす。
