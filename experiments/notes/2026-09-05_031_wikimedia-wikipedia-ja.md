# 実験031：日本語Wikipediaの追加取得と抽出

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験029でFineWeb混合コーパスを2,500 stepまで学習した結果、validation lossは改善しましたが、固定会話testの内容対応と生成品質はまだ弱い状態でした。次の20M級モデルでデータ不足による過学習を避け、会話・医療以外の一般知識も補うため、日本語Wikipediaを追加source候補として取得します。

今回の仮説は、Wikimedia Wikipediaの百科事典本文を少量でも追加すれば、既存のFineWeb・青空文庫・会話・医師国家試験の混合データに、固有名詞や説明文の分布を補えるというものです。ただしWikipediaは会話文ではないため、固定chat-testの改善は期待せず、domain別validationと後続のsource ablationで効果を切り分けます。まず全15シャードを取得せず、再現可能な一つのシャードだけを用い、ディスク使用量と処理時間を抑えます。

## 取得元、ライセンス、完全性

取得元は`wikimedia/wikipedia`の`20231101.ja` subsetです。Dataset revisionは`b8e579a0c09383e0e254c9980d56833d16048707`、入力は`train-00000-of-00015.parquet`、元データはHugging Face配布元とWikimedia Dumpsに記録されています。ライセンス表示はCC BY-SA 3.0およびGFDLです。入力ファイルは`data/downloads/wikimedia-wikipedia-20231101-ja/`へ保存し、`.gitignore`の対象としてGitHubへは追加しません。

入力ファイルのサイズは611,504,422 bytes、SHA-256は`4751c14478e712fd637bd83c2cf3537b0e299ea5115e9a78ddededf42f34c29d`です。parquetの行数は92,632で、列は`id`、`url`、`title`、`text`でした。元のparquet、元のSQLite、既存のFineWebファイルは変更していません。

## 抽出方法

`scripts/import_wikimedia_wikipedia_ja.py`を追加し、記事の`title`と`text`を読み込みます。本文がタイトルで始まらない場合だけタイトルを先頭へ補い、Unicode NFKCと空白を正規化し、一記事一行のUTF-8本文へ変換します。空記事と同一本文の重複は除外し、`--max-tokens`指定時には既存TokenizerのToken数にEOS 1個を加えて上限内で停止します。入力行を任意に捨てる既定値は0とし、重複や抽出数をmanifestへ記録します。

本抽出では既存Tokenizer`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`を使い、最大5,000,000 Tokenを上限にします。出力は`artifacts/corpus/wikimedia-wikipedia-ja-v1.txt`、manifestは`artifacts/corpus/wikimedia-wikipedia-ja-v1.manifest.json`です。これはデータ追加そのものの実験であり、20M本学習とは別の再現可能な前処理段階として記録します。

## 実行コマンド

```bash
.venv/bin/python scripts/import_wikimedia_wikipedia_ja.py \
  --input data/downloads/wikimedia-wikipedia-20231101-ja/train-00000-of-00015.parquet \
  --output artifacts/corpus/wikimedia-wikipedia-ja-v1.txt \
  --manifest artifacts/corpus/wikimedia-wikipedia-ja-v1.manifest.json \
  --max-tokens 5000000 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model
```

## 実験中の記録

2026-09-05、最初のWikipediaシャードを取得し、サイズとSHA-256を確認しました。importerと小さなparquet fixtureを用いた回帰テストを追加し、回帰テストは53 passed、3 skippedとなりました。実データの抽出を同日実行し、約2.59秒で完了しました。

## 結果と解釈

実データから738記事を抽出し、Token数は4,981,535でした。入力92,632行のうち、Token予算へ到達した739行目までを走査し、空記事0、重複除去0で停止しました。既存TokenizerのToken数にEOSを加えて数えているため、目標5,000,000 Tokenに対して4,981,535 Tokenとなっています。出力は15,401,706 bytes、SHA-256は`b0b5e3389db4e31e428413637c89b1e1173e18d6ec755bfc3c4a3f51d3f6a409`、manifestは`d62ab18dc1bc8f50004053fbb82652f6926c73a9897a3f0180762bef243e5fb1`です。

抽出した本文は`artifacts/corpus/wikimedia-wikipedia-ja-v1.txt`、条件と完全性情報は`artifacts/corpus/wikimedia-wikipedia-ja-v1.manifest.json`に保存しました。本文の内容はGitへ追加せず、manifestをGitHubへ記録することで、ローカルの大きなデータを無制限に増やさず再取得・再検証できるようにしています。取得済みparquetは`data/downloads/wikimedia-wikipedia-20231101-ja/train-00000-of-00015.parquet`に残し、入力hashとrevisionをmanifestへ記録しています。

今回の抽出処理自体は成功しました。ただし一つのシャードの先頭部分だけなので、Wikipedia全体の代表性はありません。また、タイトル補完と空白の正規化により原文の段落境界は保持していません。次の比較では、既存5M混合コーパスへこのsourceを追加した条件と、追加しない条件を同じTokenizer・同じ学習stepで評価し、Wikipediaの追加を単独の改善と誤認しないようにします。

## 次に試すこと

抽出したWikipedia本文を既存5M混合コーパスへ追加したToken予算版を作り、まず同一5Mモデル・同一2,500 stepでdomain validationを比較します。その後、実験030の20Mモデルへ進みます。Wikipedia追加で会話評価が悪化した場合は、一般知識sourceと会話sourceを別々に学習するsource ablationへ戻します。
