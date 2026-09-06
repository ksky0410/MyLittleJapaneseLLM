# 実験109：未使用日本語会話・医療データを含む20M追加事前学習

## 実施前の計画

- 実施日：2026-09-07
- 担当：Codex
- 状態：データ準備前
- 使用ブランチ：`main`

実験105〜108では、50Mモデルの一般会話の自然さは改善したが、医師国家試験の正解率は17〜19%程度で頭打ちになった。実験104以降の事前学習ではFineWeb2とWikipediaを中心に40M級のToken列を使ったが、元の会話・医療コーパスは全量ではなく、初期の混合データから一部だけが採用されている。そこで今回は、既存の日本語会話・医療問題・青空文庫コーパスから、これまでの混合列に採用されていない単位を含む追加コーパスを作り、既存の20M混合コーパスをreplayとして残しながら、合計約20M tokensを追加事前学習する。

### 仮説

会話と医療の実データを事前学習段階でも広く見せることで、一般会話の文脈適合と医療問題の語彙・選択肢表現が改善する可能性がある。既存20M混合コーパスをreplayとして含めるため、完全な新規データだけを使うよりも、FineWeb・Wikipedia由来の日本語能力を維持しやすいと予想する。ただし、SFT済みcheckpointへraw next-token pretrainingを行うため、会話の話者markerやEOS形式を忘れる可能性がある。その場合は、この実験を知識側checkpointとして保存し、元のSFTを後段に再実施して回復できるかを別途確認する。

### データ設計

入力候補は次の4系統である。

1. 既存20M混合コーパス `mixed-ja-token-budget-fineweb2-wikipedia-20m-v1.txt` をreplay
2. 会話全量 `conversation-v1/train.txt`
3. 医師国家試験全量 `medical-qb-v2/train.txt`
4. 青空文庫一般文 `aozora-general-v1.txt`

`mix_corpora.py`で重複する論理単位を除外し、Tokenizer上限20,000,000 tokens、seed 10901、重みはreplay 1.0、conversation 1.0、medical 0.75、aozora 0.25で決定的に混ぜる。会話・医療の全量が20Mに満たない場合は実際に採用できた量を優先し、manifestに残す。過去の元データ、`medilink_analysis`内の原本、既存Token列は変更しない。

### 開始前の条件

- 初期checkpoint：実験105の最良 `artifacts/checkpoints/issue1-balanced-pretrain-general-medical-sft-runpod-8k/best.pt`
- 初期checkpoint SHA-256：`1652603515b24e0538abeba01a63c53da1af4de87b51738b90acebe7326b9149`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- Tokenizer：`mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- 事前学習設定：`configs/issue1-50m-pretrain-new-japanese-20m-runpod-10k.toml`
- 事前学習設定SHA-256：`7972b5d11dc132c5660b72bafe82e84626b4787070812cc636007f395048fe3e`
- 学習コード：`scripts/train_torch.py`、SHA-256 `4695dfea5487fb7d912db762c0825a524aa921247dbfb670d74b5002cc4fe001`
- 学習：batch size 8、10,000 step、総提示Token約20.48M、AdamW、learning rate 1e-5から1e-6、warmup 500、weight decay 0.1、seed 109
- validation：`fineweb2-edu-japanese-v1-test.bin`、20 evaluation batches

### 成功判定

実験105の初期値からNaN・OOM・shape errorなく10,000 stepを完走し、500 step間隔のmetricsと生成文を保存する。性能面では、FineWeb validation lossが初期値から下がり、raw固定prompt「今日はなにをしていましたか？」の空応答・反復・文脈逸脱が減ることを第一の成功条件とする。SFT形式のEOSが崩れても、それだけで実験失敗とはせず、事前学習後の再SFTで回復可能かを別に評価する。

### 実行コマンド

Runpod Pod `j9c46julmtbcb4` のA40上で、PyTorch CUDA版を使う。学習開始前に新しいToken列、設定、初期checkpointのSHA-256をRunpod上で照合する。

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_torch.py \
  --config configs/issue1-50m-pretrain-new-japanese-20m-runpod-10k.toml \
  --initial-checkpoint artifacts/checkpoints/issue1-balanced-pretrain-general-medical-sft-runpod-8k/best.pt \
  --device cuda
```

## データ準備中の記録

データ混合は完了した。入力単位は51,761件、重複除去後のunique単位は44,150件で、20,000,000 tokensを採用した。出力本文のSHA-256は `e40a6181e864abced2385be88128bda1462c310e3f5345e05f453ab3dd10a3a4`、mix manifestのSHA-256は `dcd71a8319859aa262e753aac3b74fb423d4361b55f565074eb38496a1980f58` である。

実際のToken比率はreplay 47.14%（9,427,150 tokens）、会話 41.54%（8,307,061 tokens）、医療 11.33%（2,265,789 tokens）だった。青空文庫はreplay側と本文単位が重複していたため、追加採用は0 tokensとなった。会話は全11,635単位のうち10,938単位、医療は全6,142単位のうち4,986単位を採用しており、初期混合データで採用された一部だけでなく、より広い範囲を学習に含められた。

Tokenizerでuint32列へ変換し、Token数は20,000,000、binaryのSHA-256は `7b9aa1968ad81cc1f695c452baea6362ed2117ef28db23d9f9e71753f75a73db`、metadataのSHA-256は `2a768f65ba1334fd6e6907d7cb4373fddd6f1e129b19b2e4de1da62a542da0e4` となった。大きな本文とbinary本体はGitへ追加せず、manifestとこのノートへ取得条件・hashを残す。

この結果は、当初の「replay 33%、会話33%、医療25%、青空文庫8%」という希望比率とは異なる。原因は会話・医療の単位が枯渇し、青空文庫がreplayと重複して全除外されたためである。実際に増えたデータ量とsource比率を優先し、予定との差は成功結果として明記する。

## 学習中の記録

ここに500 stepごとのvalidation loss、learning rate、elapsed time、GPUメモリ、固定prompt生成、停止理由を追記する。崩れた生成も削除しない。

### 2026-09-07：step 500

Runpod Pod `j9c46julmtbcb4` のA40上で、実験105の最良checkpointから追加事前学習を開始した。step 1のFineWeb validation lossは2.921104、step 500は2.865372で、step 500の学習率は1.0e-5、経過時間は31.34秒だった。NaN、OOM、shape errorは発生していない。学習中の生成、metrics、checkpoint metadataは専用出力先へ保存する。

### 2026-09-07：step 1,000〜10,000

validation lossはstep 1,000で2.856815、2,000で2.849935、3,000で2.848086、4,000で2.846685、5,000で2.846276、6,000で2.845728、7,000で2.845544、8,000で2.843922、9,000で2.843931、9,500で2.843884となった。step 5,500では2.846718へ一時的に悪化したが、その後に最良値を更新した。最終step 10,000までNaN、OOM、shape error、途中停止はなく、学習時間は621.30秒だった。最良checkpointはstep 9,500である。

FineWeb lossは実験105のSFT後2.921111から大きく改善した。これは追加20M tokensと10,000 stepの事前学習がvalidation上の日本語適合を改善したことを示すが、会話の自然さや医療正答率は生成評価を終えるまで判断しない。

## 実験終了後の記録

### 2026-09-07：学習・raw評価完了

10,000 stepまで学習し、FineWeb validation lossが最も低かったstep 9,500のcheckpointをbestとして採用した。best checkpointのSHA-256は `0ceaeedef9d8ab8039861078bce673977a78a8bf7d329ff57096184b9f531cea`、最良FineWeb lossは2.843884、perplexityは17.1824、学習時間は621.30秒だった。最終step 10,000のvalidation lossは2.84402付近である。NaN、OOM、shape error、途中停止はなかった。

同じ20 evaluation batchesで評価した領域別lossはFineWeb 2.843884、general 4.126995、conversation 1.969749、medical 1.847450となった。実験105のFineWeb 2.921111、conversation 2.070219、medical 1.977347から改善した一方、generalは4.084512から悪化した。会話・医療のraw validation lossが大きく下がったため、追加コーパスの効果は確認できたが、SFT形式の能力とは区別する必要がある。

固定一般会話48例ではEOS 48/48、平均生成9.29 tokens、token overlap F1 0.1252だった。実験108のF1 0.2359から大きく悪化した。医療162例ではEOS 157/162、平均生成23.20 tokens、token overlap F1 0.0905となり、「正解は…です。」を抽出できたのは12例、正解は2例、正解率1.23%だった。raw next-token事前学習が会話SFTの返答形式と医療回答形式を忘れさせたことが明確である。

したがって、実験109は総合モデルとしては採用しないが、追加20M tokensと広い会話・医療データで知識側validationを改善できる有用なpretraining checkpointとして保存する。次はこのbest checkpointを初期値に、実験105と同じ一般・医療SFTを再適用する。再SFT後に会話・医療正答率が回復し、FineWeb改善も一部維持できれば、今後の標準パイプラインを「追加事前学習→十分なSFT」とする根拠になる。回復しなければ、追加事前学習のlearning rateまたはSFTデータ量を見直す。

学習中のmetrics、summary、checkpoint metadata、step別生成は `artifacts/checkpoints/issue1-50m-pretrain-new-japanese-20m-runpod-10k/` と `artifacts/samples/issue1-50m-pretrain-new-japanese-20m-runpod-10k/` に、評価JSON/TXTは `artifacts/evaluations/exp109/` に保存する。
