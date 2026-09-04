# 実験ノート：青空文庫の日本語長文を使った学習

## 基本情報

- 実験番号：004
- 計画作成日：2026-09-05
- 担当者：ユーザーとCodex
- Gitコミット：401a07bを基準に、データ取り込み機能を追加して実施する予定
- ブランチ：main
- 状態：正式作品の取り込みを完了。Token化・学習・評価は未実施

このノートは実験前の計画と、正式作品を取り込むまでの記録である。取得した元zipと生成物はリポジトリ外または`artifacts/`配下に置き、公開リポジトリへ大きなデータを含めない。

## データの出所と注意

対象作品は、青空文庫の図書カード「吾輩は猫である」である。

- 図書カード：https://www.aozora.gr.jp/cards/000148/card789.html
- 取得ファイル：https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip
- 取得時点：2026-09-05
- 元ファイル：`789_ruby_5639.zip` に含まれる `wagahaiwa_nekodearu.txt`
- 図書カードに記載された文字コード：JIS X 0208／ShiftJIS
- 取得したzipのSHA-256：`6545750b89ee2c57f215a65079eeab56ee7c997373b7b86e5b00ae74fe69208f`

作品中には、現在の基準では不適切と受け取られる可能性がある表現が含まれる旨を図書カードが注意している。学習・公開・生成結果の扱いでは、その注意をREADMEと実験結果に残し、生成文を一般的な日本語能力の代表例として扱わない。

## 今回確かめたいこと

短い内蔵サンプルではなく、まとまった日本語の文体と文脈を含む長文を使うと、モデルのvalidation lossと固定prompt生成がどの程度変わるかを確認する。まずはデータ取得、文字コード変換、青空文庫特有のルビ除去、train/validation分割、Tokenizer学習までを再現可能にする。

## 仮説

長文の文脈と文章パターンを増やすことで、内蔵サンプルだけで学習したモデルより、助詞、句読点、文末表現の連続が安定すると予想する。ただし、単一作品の文体に強く偏るため、一般的な日本語能力ではなく、作品文体の学習として評価する。

## 予定している処理

1. `scripts/import_aozora.py`でShift_JISのzipまたはtxtをUTF-8のテキストへ変換する。
2. 本文開始後の末尾256行から、行頭の`底本：`・`入力：`・`校正：`・`青空文庫作成ファイル：`などの後方候補を探し、フッターの先頭から作品情報や入力・校正情報を除く。本文直後の区切り線も除く。
3. ルビ表記を本文だけに変換し、本文の改変内容を記録する。
4. 行単位で処理し、空行を除く。4,000文字を超える本文行は分割する。
5. manifestへ入力・出力のSHA-256、文字数、行数、除去行数、注記数、分割数を保存する。
6. seedを固定してtrain/validationへ分割する。
7. SentencePiece Unigramを学習し、Token数と分割例を記録する。
8. 小型モデルで短時間の学習を行い、validation loss、perplexity、固定prompt生成を保存する。

## モデルと成功基準

最初は5Mパラメータ前後のモデルを候補とし、実際のTokenizer語彙数を確定してからパラメータ数を記録する。学習step、context length、batch size、学習率は、MacBook上で完走できるdebug条件から段階的に増やす。

データ変換からToken化までがエラーなく再現でき、出所・加工履歴・ハッシュがmanifestとこのノートに残ることを第一の成功条件とする。その後、学習が完走し、学習前後のlossと固定promptの出力を比較できれば、長文コーパスの導入実験として成功とする。

## 実行予定のコマンド

データ取り込み機能が完成した後、実際に使用したコマンドをここへ追記する。取得済みの元zipを再取得する場合も、ファイルのSHA-256が一致することを確認する。

最初の実装確認では、実データをダウンロードせず、テストが一時生成するShift_JIS zipだけを使用した。

```bash
.venv/bin/python -m pytest -q tests/test_aozora_import.py
.venv/bin/ruff check scripts/import_aozora.py tests/test_aozora_import.py
```

実際には全体確認も追加で実施し、次の結果になった。

```text
.venv/bin/ruff check .       -> All checks passed!
.venv/bin/python -m pytest -q -> 11 passed in 0.35s
既存スクリプトを含む --help 確認 -> all script help checks passed
```

テストでは、小さなShift_JIS zipと単独のShift_JIS txtを一時生成した。zip内txtの選択、区切り線あり・なしのフッター除去、UTF-8出力、`［＃注記］`の除去、`｜猫《ねこ》`の本文化、最大文字数での分割、入力・出力SHA-256、文字数・行数・除去数・分割数、既定manifest名を確認した。

実作品での回帰検証により、本文直後に区切り線がない場合のフッター残留が判明したため、本文開始後の末尾256行にある行頭一致のフッター候補を後方クラスタから検出する処理へ修正した。修正後に同じ実作品を再取り込みし、フッターの残留がないことを確認した。

修正後の正式な取り込みには、次のコマンドを使用した。

```bash
.venv/bin/python scripts/import_aozora.py \
  --input /tmp/my-little-japanese-llm-data/789_ruby_5639.zip \
  --output artifacts/corpus/aozora-neko-formal-v2.txt \
  --manifest artifacts/corpus/aozora-neko-formal-v2.manifest.json \
  --source https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip \
  --max-chars 4000
```

## 結果

正式な作品zipのimportは完了した。Token化、学習、評価、生成は未実施である。

ただし、変換方針を確認するための予備検証を行った。取得したzipを一時領域でShift_JISからUTF-8へ変換し、作品情報とルビを除いた本文は2,255行、960,407 byteになった。既存の`prepare_data.py`でseed 42、validation ratio 0.05に分割すると、学習2,142行・302,876文字、検証113行・16,442文字になった。

そのままSentencePieceへ渡す予備検証では、要求語彙数512が実効語彙数3,013へ自動調整された。一方、4,192文字を超える39行がSentencePieceによって学習対象から除外された。これはデータを意図せず捨てるため正式な実験には不適切であり、予備検証は未完了扱いとする。正式なimport処理では、長い段落をTokenizerの最大長以下へ分割し、分割件数をmanifestへ記録してから学習する。
そのままSentencePieceへ渡す予備検証では、要求語彙数512が実効語彙数3,013へ自動調整された。一方、4,192文字を超える39行がSentencePieceによって学習対象から除外された。これはデータを意図せず捨てるため正式な実験には不適切であり、予備検証は未完了扱いとした。

正式なimportでは、同じzipから入力文字数377,325、入力2,376行の本文候補を読み込み、注記461件、ルビ9,214件を処理した。空行25行、注記行60行など合計120行を除去し、長文2段落を3区間へ分割した。出力は2,259行、320,716文字で、4,000文字を超える行は0行だった。出力末尾には作品の終端が残り、`［＃注記］`、`《ルビ》`、および作品情報フッターの残留は確認されなかった。詳細な件数とSHA-256は`artifacts/corpus/aozora-neko-formal-v2.manifest.json`に保存した。

この予備検証で生成したデータとTokenizerは`artifacts/`配下に置いており、Git管理していない。予備検証の結果は、データ取り込み機能の要件を具体化するためにのみ利用する。

## 次に試すこと

正式な出力を`prepare_data.py`でtrain/validationへ分割し、SentencePiece Unigramの実効語彙数、Token数、分割例を記録する。その後、約5Mパラメータのモデルを短時間学習し、validation loss、perplexity、固定promptの生成結果を比較する。
