# 実験ノート：一般日本語sourceの青空文庫作品追加

## 基本情報

- 実験番号：011
- 記録日：2026-09-05
- 担当者：ユーザーとCodex
- 実験開始時のGitコミット：`065df14`
- ブランチ：`main`
- 状態：取得前

## 今回確かめたいこと

実験008・009では一般sourceが『吾輩は猫である。』一作品だけだったため、単位数80%を維持しようとすると2,680単位程度しか作れなかった。今回は青空文庫の別作品を追加し、一般sourceの種類と量を増やす。仮説は、複数作品を文書単位で追加すれば、会話・医療を水増しせずに一般80%を維持でき、作品固有の反復を減らせるというものである。

追加対象は、公式図書カードから取得先を固定できる次の3作品とする。

- 『坊っちゃん』：図書カード`https://www.aozora.gr.jp/cards/000148/card752.html`、zip`https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip`
- 『こころ』：図書カード`https://www.aozora.gr.jp/cards/000148/card773.html`、zip`https://www.aozora.gr.jp/cards/000148/files/773_ruby_5968.zip`
- 『それから』：図書カード`https://www.aozora.gr.jp/cards/000148/card1746.html`、zip`https://www.aozora.gr.jp/cards/000148/files/1746_ruby_18324.zip`

既存の『吾輩は猫である。』を含めた4作品を一般sourceとし、既存の会話・医療trainを加えた比較用コーパスを作る。元zipと変換本文はGitへ追加せず、取得日時・URL・SHA-256・各作品manifestを記録する。元の`medilink_analysis`ディレクトリは使用せず、変更しない。

## 実行前の条件

ダウンロード先は`/tmp/my-little-japanese-llm-data/aozora-general-v1/`とする。既存ImporterでShift_JISのzipを読み、ルビ・注記・ヘッダー・フッターを除いたUTF-8本文をsmall_llm側の次の作業ディレクトリへ出力する。

```text
artifacts/corpus/aozora-general-v1/bocchan/corpus.txt
artifacts/corpus/aozora-general-v1/bocchan/manifest.json
artifacts/corpus/aozora-general-v1/kokoro/corpus.txt
artifacts/corpus/aozora-general-v1/kokoro/manifest.json
artifacts/corpus/aozora-general-v1/sorekara/corpus.txt
artifacts/corpus/aozora-general-v1/sorekara/manifest.json
```

各作品を一論理単位として混合し、作品間の重複を除く。既存Nekoのtrainを含めた一般sourceの作成では、会話・医療と同じ`mix_corpora.py`を使うが、一般作品は同じsource名へまとめる前に作品別manifestを残す。validationへの作品混入を避けるため、当面は既存Nekoのvalidationを一般評価へ使用し、追加作品の本文はtrain側へ置く。

## 成功条件

3 zipが取得でき、各入力のSHA-256と作品別出力manifestが保存されること。本文にルビ記号、Aozora注記、底本・入力者情報が残らないこと。作品ごとに0行でなく、長文のskipがないこと。元データディレクトリに変更がないことを確認する。失敗した作品は削除せず、原因とともに記録する。

## 結果

3 zipの取得と変換は成功した。取得したzipのSHA-256は、坊っちゃんが`53c4d86033d0e4590c0354df5fec2709820ef21c09fa402826ff33376ec21228`、こころが`c55d7bf6c7cc5bd960ce291949f38cae9ba6a672a6142c3b9e1a9f5a159fa4c4`、それからが`d7fa3626fe64a374bd50a00e6e25c6f0ade1fc080a2128b69e79d39266f00222`である。変換後の本文は、坊っちゃん482行・88,773文字、こころ1,331行・161,572文字、それから1,910行・172,220文字となった。各作品の入力・出力SHA-256、除去行数、注記数、ルビ数は作品別manifestへ保存した。

3作品を既存の吾輩は猫である。trainと一緒に、seed 42・同じsource優先順で混ぜた。入力5,869単位から完全一致重複111単位を除き、一般sourceは5,758単位、727,095文字となった。作品別の採用単位比率は、吾輩36.97%、坊っちゃん8.09%、こころ21.80%、それから33.14%であり、一作品だけに偏った状態を緩和できた。一般sourceのmanifestは`artifacts/corpus/aozora-general-v1.manifest.json`に保存した。

元zipのダウンロード先は`/tmp/my-little-japanese-llm-data/aozora-general-v1/`で、small_llm側には変換本文と軽量manifestだけを置いた。元の`/Users/koseki/projects/medilink_analysis`は変更されておらず、Git状態は`feature/quizpack-v1`のままである。本文に残るルビ記号やAozora注記を確認したが、Importerの変換後manifest上、各作品は正常に本文を出力している。

この結果から、一般sourceを単位数5,758まで増やせたため、会話・医療を水増しせずにtarget 7,000、重み8:1:1で一般5,600・会話700・医療700を選べる見込みである。次の実験012でこの混合比を作り、Tokenizer後のsource別token比率も確認する。取得・変換結果の確認だけであり、モデル学習はまだ実施していない。
