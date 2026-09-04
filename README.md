# My Little Japanese LLM

Apple SiliconのMacBook上で、日本語Tokenizerと小さなdecoder-only Transformerをゼロから動かして学ぶための最小プロジェクトです。外部から巨大なデータセットを自動取得せず、内蔵サンプルか、ユーザーが指定したUTF-8テキストだけで一連の処理を実行できます。

このリポジトリには、コーパスの準備、SentencePiece Unigram/BPEの学習、Token化、MLXモデルの学習、validation lossの評価、checkpointの保存、固定promptの生成、モデル形状の確認が実装されています。実験の計画と結果は [AGENTS.md](AGENTS.md) の規則に従い、`experiments/notes/` に記録してください。

## 動作環境とインストール

本学習はApple Silicon MacとPython 3.11以上を想定し、MLXを使用します。プロジェクトのコマンドは、現在のシェルのPATHに依存しないよう `.venv/bin/python` を直接指定します。

```bash
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[apple,dev]"
```

MLXを使わずにコーパス処理、Tokenizer、設定、モデル形状、テストだけを確認する場合は、次のようにインストールできます。

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

SentencePieceやMLXがない場合、Tokenizerまたは学習系の実行時に、必要なインストールコマンドを含むエラーを表示します。`--help`、設定読み込み、corpus処理、モデル規模の概算はMLXを遅延importするため、MLXなしでも実行できます。

## 内蔵サンプルでdebugする

以下はリポジトリのルート以外から実行しても、相対パスをリポジトリルート基準で解決します。まず、内蔵サンプルを正規化し、seed 42で文書単位に決定的分割します。

```bash
.venv/bin/python scripts/prepare_data.py \
  --input data/sample_ja.txt \
  --output-dir artifacts/corpus \
  --val-ratio 0.2 \
  --seed 42
```

次に、学習splitからTokenizerを作ります。小さなコーパスで要求語彙数が必要な文字種類数より小さい場合は、スクリプトがSentencePieceに渡す語彙数を自動的に下限まで引き上げます。反対に、利用可能なpieceが少ない場合は `hard_vocab_limit=False` により実際の語彙数が要求値を下回ることがあります。実際の語彙数はコマンドの出力で確認してください。

```bash
.venv/bin/python scripts/train_tokenizer.py \
  --input artifacts/corpus/train.txt \
  --model-prefix artifacts/tokenizer/ja \
  --vocab-size 128 \
  --model-type unigram
```

SentencePieceの`max_sentence_length`は既定値を4,192のまま維持していますが、日本語では1文字がUTF-8で複数バイトになるため、文字数が4,192未満でも長い文が除外されることがあります。実データでは、必要な最大バイト長を見積もって`--max-sentence-length`を指定してください。例えば、正式な青空文庫コーパスでは次のように20,000を指定できます。

```bash
.venv/bin/python scripts/train_tokenizer.py \
  --input artifacts/corpus/aozora-neko-formal-v2/train.txt \
  --model-prefix artifacts/tokenizer/aozora-neko-formal-v2-unigram \
  --vocab-size 4096 \
  --model-type unigram \
  --max-sentence-length 20000
```

`--max-sentence-length`は1以上の整数で指定します。値はSentencePieceへ渡すUTF-8バイト長の上限です。APIからは`train_sentencepiece(..., max_sentence_length=20000)`として指定できます。

学習splitと検証splitは別々にToken化します。splitを一つのファイルへ連結しないため、train/validationの境界が混ざりません。各binaryには、Token数、語彙数、EOS ID、SHA-256を記録したJSONも保存されます。

```bash
.venv/bin/python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/ja.model \
  --input artifacts/corpus/train.txt \
  --output artifacts/tokens/train.bin

.venv/bin/python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/ja.model \
  --input artifacts/corpus/val.txt \
  --output artifacts/tokens/val.bin
```

UnigramとBPEを同じ入力で比較するときは、Tokenizerを複数回指定してJSONレポートを出力します。レポートには実語彙数、総token数、平均文字/token、固定した3つの日本語サンプルのpiece分割が含まれます。

```bash
.venv/bin/python scripts/tokenizer_report.py \
  --tokenizer artifacts/tokenizer/unigram.model \
  --tokenizer artifacts/tokenizer/bpe.model \
  --input artifacts/corpus/train.txt
```

UnigramとBPEを同じモデル条件で短く学習して比較する場合は、Token列とcheckpointの保存先を分けた設定を使います。先に両Tokenizerと4つのToken列を作り、その後に2つの学習を実行します。

```bash
.venv/bin/python scripts/train_tokenizer.py --input artifacts/corpus/train.txt --model-prefix artifacts/tokenizer/unigram --vocab-size 128 --model-type unigram
.venv/bin/python scripts/train_tokenizer.py --input artifacts/corpus/train.txt --model-prefix artifacts/tokenizer/bpe --vocab-size 128 --model-type bpe
.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/unigram.model --input artifacts/corpus/train.txt --output artifacts/tokens/unigram-train.bin
.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/unigram.model --input artifacts/corpus/val.txt --output artifacts/tokens/unigram-val.bin
.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/bpe.model --input artifacts/corpus/train.txt --output artifacts/tokens/bpe-train.bin
.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/bpe.model --input artifacts/corpus/val.txt --output artifacts/tokens/bpe-val.bin

.venv/bin/python scripts/train.py --config configs/debug-unigram.toml
.venv/bin/python scripts/train.py --config configs/debug-bpe.toml
```

Unigram側の評価には`configs/debug-unigram.toml`と`artifacts/checkpoints/unigram/step_000100.npz`を、BPE側の評価には`configs/debug-bpe.toml`と`artifacts/checkpoints/bpe/step_000100.npz`を指定します。比較の計画と結果は [実験003のノート](experiments/notes/2026-09-05_003-tokenizer-training.md) に記録します。

学習前に、Tokenizerの実際の語彙数とモデルの入力・出力形状、概算パラメータ数を確認します。

```bash
.venv/bin/python scripts/inspect_model.py --config configs/debug.toml
```

実験ノートに予定を記録した後、debug学習を実行します。`configs/debug.toml` はdim 64、2層、context length 64、100 stepの小さな設定です。学習中はstep 1と20 stepごと、または最終stepにtrain loss、validation loss、perplexity、学習率をJSONLと標準出力へ記録します。ログ間隔は1000 stepを超えないようにしています。

```bash
.venv/bin/python scripts/train.py --config configs/debug.toml
```

学習によって次の成果物が作られます。

- `artifacts/checkpoints/step_000100.npz` と対応するmetadata JSON
- `artifacts/checkpoints/metrics.jsonl`
- `artifacts/checkpoints/summary.json`
- `artifacts/samples/step_000000.txt` と学習中の固定prompt生成結果

checkpointは重みファイルとJSON metadataに分けて保存します。生成と評価では、metadataのformat、モデルの語彙数・dim・層数・head数・context length・MLP倍率を現在の設定およびTokenizerと比較し、不一致ならロード前に拒否します。

```bash
.venv/bin/python scripts/evaluate.py \
  --config configs/debug.toml \
  --checkpoint artifacts/checkpoints/step_000100.npz

.venv/bin/python scripts/generate.py \
  --config configs/debug.toml \
  --checkpoint artifacts/checkpoints/step_000100.npz \
  --output artifacts/samples/manual.txt
```

## 手元のコーパスを使う

巨大なデータセットのダウンロードは行いません。手元のUTF-8テキストを指定して、同じコマンドを実行してください。入力は1行を1文または1段落として扱います。

```bash
.venv/bin/python scripts/prepare_data.py \
  --input /path/to/your/japanese.txt \
  --output-dir artifacts/my-corpus \
  --val-ratio 0.05 \
  --seed 42
```

その後、Tokenizerとencodeの `--input` を `artifacts/my-corpus/train.txt` と `artifacts/my-corpus/val.txt` に変更します。データの出所、ライセンス、前処理、ハッシュ、Token数は実験ノートへ記録してください。

## 青空文庫のShift_JIS作品を取り込む

青空文庫のzipまたはtxtは、作品ヘッダーとフッター、`［＃注記］`形式の注記を取り除き、`｜親文字《ルビ》`および親文字に直接付いた`《ルビ》`を親文字だけの本文へ変換できます。フッターは、本文開始後の末尾256行から、行頭が`底本：`・`入力：`・`校正：`・`青空文庫作成ファイル：`などに一致する後方の候補まとまりを探し、その先頭を本文の終端とします。本文直後に区切り線がある形式では、その区切り線も除去します。長い本文行は、SentencePieceが捨てないように既定の4,000文字ごとへ分割します。入力ファイル自体はGitへ追加せず、出所URLとmanifestのハッシュを実験ノートへ残してください。

```bash
.venv/bin/python scripts/import_aozora.py \
  --input /path/to/aozora作品.zip \
  --output artifacts/corpus/aozora-neko.txt \
  --manifest artifacts/corpus/aozora-neko.manifest.json \
  --source https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip
```

入力の既定文字コードは`shift_jis`です。作品によって異なる場合は`--encoding`で変更できます。manifestには入力・出力のSHA-256、文字数、行数、除去した行と注記の件数、ルビ件数、長文の分割件数、入力zip内のtxt名を保存します。変換後は`prepare_data.py`へUTF-8の出力を渡してください。

正式Token列で5,000 stepの独立実験を行う場合は、`configs/aozora-5m-full.toml`を`train.py`の入口として使用します。checkpointとsampleは`aozora-5m-full`専用ディレクトリへ保存されます。

## 医師国家試験SQLiteを取り込む

`/Users/koseki/projects/medilink_analysis/data/qb.sqlite`を読み取り専用で開き、`questions`と`descriptions`から医師国家試験データセットを作成できます。元のSQLiteへ書き込まず、出力はsmall_llm側の`artifacts/corpus/medical-qb-v1/`だけに保存します。119回をvalidation、120回をtest、700回をchallenge、その他をtrainへ分け、説明JSONがない問題も説明欄を空にして処理を続けます。

```bash
.venv/bin/python scripts/import_medical_qb.py \
  --input /Users/koseki/projects/medilink_analysis/data/qb.sqlite \
  --output-dir artifacts/corpus/medical-qb-v1
```

分割回を変更する場合は、`--validation-version 118 --test-version 119 --challenge-version 701`のように指定します。各オプションは複数回指定でき、同じexam_versionをvalidation・test・challengeへ重複指定するとエラーになります。出力には構造化された`train.jsonl`・`validation.jsonl`・`test.jsonl`・`challenge.jsonl`と、問題・選択肢・正解・ポイント・選択肢解説を自然なラベルで連結した1問1行の`train.txt`・`validation.txt`・`test.txt`・`challenge.txt`が含まれます。画像URLは保存せず、`[図表あり]`へ置き換えます。件数、exam_version別件数、欠損件数、画像件数、入力・出力SHA-256はmanifestへ記録します。

## 会話コーパスを安全に取り込む

公開元のREADME・LICENSE・利用条件を確認したうえで、`/tmp`へ取得したRealPersonaChatとMRMPを会話単位で取り込めます。既定のライセンス表記はCC BY-SA 4.0です。会話から個人を特定しようとせず、実在の話者になりすます用途にも使わないでください。出力には公開元の利用条件と、この安全上の注意をmanifestへ残します。

```bash
.venv/bin/python scripts/import_conversations.py \
  --input /tmp/real-persona-chat \
  --input /tmp/multi-relational-multi-party-chat-corpus \
  --source-name real-persona-chat \
  --source-name mrmp \
  --source-url https://github.com/nu-dialogue/real-persona-chat \
  --source-url https://github.com/nu-dialogue/multi-relational-multi-party-chat-corpus \
  --license 'CC BY-SA 4.0' \
  --output-dir artifacts/corpus/conversation-v1 \
  --seed 42
```

## 一般日本語・医療・会話を混ぜる

各入力はUTF-8の1行1文、1問題、または1会話テキストとして扱います。`mix_corpora.py`は各sourceをseed付きでshuffleし、`--target-units`で指定した総単位数を重みの比率で抽出します。たとえば重みを8・1・1、targetを10,000とすれば、一般日本語・会話・医療をおおむね80%・10%・10%で混ぜられます。sourceが先に尽きた場合は、残りを他のsourceへ再配分します。単位の複製は行わず、会話・問題・段落の途中でも分割しません。

```bash
.venv/bin/python scripts/mix_corpora.py \
  --source general=artifacts/corpus/aozora-neko-formal-v2/train.txt \
  --source medical=artifacts/corpus/medical-qb-v2/train.txt \
  --source conversation=artifacts/corpus/conversation-v1/train.txt \
  --weight general=8.0 \
  --weight medical=1.0 \
  --weight conversation=1.0 \
  --target-units 10000 \
  --seed 42 \
  --output artifacts/corpus/mixed-ja-pretraining.txt \
  --manifest artifacts/corpus/mixed-ja-pretraining.manifest.json
```

manifestには各sourceの入力SHA-256、行数、文字数、重み、希望比率、採用単位数・文字数、重複除去数と、混合出力のSHA-256を記録します。異なるsourceに同じ本文が元からある場合は、source指定順で最初の一つだけを採用します。

## テストと入口の確認

プロジェクトの仮想環境を使い、軽量なテストを実行できます。MLXまたはSentencePieceがない環境では、それらを必要とするテストだけがskipされます。

```bash
.venv/bin/python -m pytest -q

for script in scripts/import_aozora.py scripts/import_medical_qb.py scripts/import_conversations.py scripts/mix_corpora.py scripts/prepare_data.py scripts/train_tokenizer.py scripts/encode_data.py scripts/tokenizer_report.py scripts/inspect_model.py scripts/train.py scripts/generate.py scripts/evaluate.py; do
  .venv/bin/python "$script" --help >/dev/null
done
```

## 実験成果物のGit管理

学習の進行をGitHubで細かく確認できるよう、`artifacts/samples/**/*.txt`に保存されるstepごとの生成文とreloaded生成文、`artifacts/checkpoints/**/metrics.jsonl`、`summary.json`、checkpoint metadataのJSON、Tokenizer・corpusのmanifestとreport JSONはGitの追跡対象として、実験の区切りごとにcommit・pushします。生成文はlossだけでは分からないモデルの変化を確認するため、悪い出力や途中の出力も削除せず残します。

一方、巨大な`.npz`・`.bin`・Tokenizerモデルや語彙ファイル、元データなどは引き続きGitへ追加しません。これらは保存場所、作成条件、SHA-256を実験ノートへ記録し、必要なバイナリをGitHubへ持ち込まずに再現できるよう管理します。
