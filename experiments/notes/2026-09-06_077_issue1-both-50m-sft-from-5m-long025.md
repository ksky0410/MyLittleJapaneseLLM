# 実験077：50M基盤における長文応答25%層化SFT

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。Issue #1で候補になっている現代的な会話データを、一般日本語と医師国家試験由来データを含む共通の日本語モデルへ組み込む方針を維持します。医療専用モデルにはせず、一般・会話・医療・RPC・MRMPを分けた評価で、データの影響を確認します。元の`/Users/koseki/projects/medilink_analysis`と医師国家試験の原データは変更・削除しません。

実験076では、50Mパラメータの事前学習済みモデルへresponse-only SFTを行い、SFTと一般日本語rehearsalを0.80対0.20で混ぜました。076は、同じ50Mモデルで追加の5M Token事前学習を経たため、20M実験より長い応答の長さ別評価が改善しましたが、短文評価は前段の50M SFTより低下しました。20Mの069〜072では、SFT部分の長文例を増やすと改善するseedもあった一方、seed 123と777では再現性が弱く、単一seedの採用判断は保留しています。

今回は076と同じ50M事前学習済みbase、同じSFTデータ、同じrehearsal Token列、同じseed・学習率・step数を固定し、SFT部分6例のうち応答24 Token以上の長文を2例にします。batch size 8、rehearsal ratio 0.20のため、実際には各SFT部分6例から長文2例・通常4例を抽出する設定です。076は長文比率を指定していないため、今回の差分は長文層化だけです。20Mで見えた長文改善が50M・5M Token基盤でも再現するか、また短文性能や一般・医療の保持を損なうかを検証します。

仮説は、076に比べてlong F1と平均生成長が改善し、MRMP・RPCの長い履歴で応答のToken overlapが上がることです。ただし、過去のseed分散を踏まえ、1回のF1差だけで長文比率を標準条件へ採用しません。5領域loss、EOS到達率、長さ別・source別F1、固定prompt生成、生成本文の質的確認を合わせて判断します。

## 再現条件

基準は実験076の評価完了commitです。設定は[`configs/issue1-both-50m-sft-from-5m-long025-3k.toml`](../../configs/issue1-both-50m-sft-from-5m-long025-3k.toml)、baseは実験075の[`best.pt` metadata](../../artifacts/checkpoints/issue1-both-50m-pretrain-5m-2p5k/)に記録されたcheckpointです。実験075のbase checkpoint SHA-256は`71931b2c689c2fbaa31c8c92c022a21fac571894ec2993a59be48644794e5e17`です。

Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`、モデルはRoPE・LayerNorm・SwiGLU、dim 576、12層、9 heads、context length 256、約50M parametersです。設定上のseedは42、batch sizeは8、最大3,000 step、learning rateは5e-5から5e-6、warmup 100、weight decay 0.01、EOS loss weight 0.50です。SFT trainは64,423例、validationは49,045例です。SFTはresponse-only loss、rehearsalは`mixed-ja-80-10-10-v2-train.bin`を使います。

Colab送信用bundleは`/tmp/small_llm-colab-077-XXXXXX.tar.gz`、236,435,505 bytes、SHA-256は`8fefb4f4b99afc8d685b83d282de6961f1caa215a99607d2d286d4569fcd90ef`です。bundleには077の実行コード、設定、075 best checkpoint、加工済みSFTデータ、rehearsal Token列、Tokenizerだけを含め、元JSONL、医師国家試験原本、`medilink_analysis`は含めていません。

使用する入力のSHA-256は次のとおりです。

- 設定ファイル：`d2f0ca2b98badcfbfbc9cbf6896d5d42afaf701e09fc35d7c8ca28880fc2af3f`
- base checkpoint：`71931b2c689c2fbaa31c8c92c022a21fac571894ec2993a59be48644794e5e17`
- SFT train：`645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`
- SFT validation：`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`
- rehearsal Token列：`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`
- Tokenizer：`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`

再現コマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-50m-sft-from-5m-long025-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-5m-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-50m-sft-from-5m-long025-3k \
  --samples-dir artifacts/samples/issue1-both-50m-sft-from-5m-long025-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --long-response-ratio 0.25 \
  --long-response-min-tokens 24 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

成功条件は、ColabまたはMPSで3,000 stepをNaN、OOM、shape errorなく完走し、step 0〜3,000の生成本文、500 stepごとのcheckpoint metadata、metrics、summary、5領域評価、固定chat-test 48例、人手レビュー用JSONを保存することです。長文比率の研究上の採否は、076との差、既存の20M seed sweep、生成本文を見て保留を含めて判断します。

## 実験中の記録

この節には、Colab CLIの試行結果、セッション状態、MPSへの切り替え、step 1・500・1,000・1,500・2,000・2,500・3,000のloss・PPL・経過時間・学習率・固定prompt生成を時系列で追記します。悪い生成、警告、途中停止も削除せず記録します。

2026年9月6日06:50台に`colab sessions`を実行し、`No active sessions found on server.`を確認しました。その後、`colab new --session exp077-both-50m-sft-long025 --gpu T4`を実行しましたが、assignment endpointがHTTP 503 `Service Unavailable`を返しました。Colabセッションは作成されず、bundle upload、入力hash検証、モデル初期化、学習stepは発生していません。過去の075・076と同じ障害として明記し、同一条件をMPSへ切り替えます。

## 実験終了後の結果と解釈

学習終了後に、実際のbackend、最良checkpoint、学習時間、5領域loss、固定chat-testのEOS・長さ・precision・recall・F1、長さ別・source別集計、生成本文の質的観察、076との差分を追記します。設定変更や失敗があった場合は、予定との差分と原因を明記します。

## 次に試すこと

077の差が大きくても小さくても、単一seedの結論にしません。差が再現しそうならseedを変えた確認、差が小さければ長文比率よりデータ形式または人手評価へ進みます。その後、50Mモデルのcontext lengthを512へ伸ばす実験、データ量を増やす実験、現代的なinstruction・蒸留手法を順番に検証します。
