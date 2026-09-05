# 実験076：約5M Tokenで事前学習した50Mモデルへのresponse-only SFT

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。Issue #1の現代的な会話データを含む日本語混合コーパスで事前学習した50Mモデルへ、response-only SFTを適用します。医師国家試験由来のデータは一般・医療領域の保持を測るrehearsalと評価へ使いますが、医療専用モデルにはしません。元の`/Users/koseki/projects/medilink_analysis`とその原データは変更・削除しません。

実験075では、約500万Tokenで事前学習した50Mモデルが、約100万Token条件の073よりgeneral・conversation・medical・RPC・MRMPのvalidation lossをすべて改善しました。一方、固定chat-test-v1のToken overlap F1は073より低く、事前学習のlanguage modeling改善だけでは会話応答の改善が確認できませんでした。今回は075のbest checkpointを初期値にして、074と同じresponse-only SFT条件を適用します。074は073の約100万Token基盤からSFTしたため、075基盤でSFT後の会話応答がどこまで変わるかを比較できます。

仮説は、より十分な日本語基盤を持つ075へSFTすると、074と同じSFT条件でも固定chat-testのoverall・medium・long F1が改善し、短い応答だけでなく話題への適合性が上がることです。反対に、SFT後もF1が伸びない場合は、現行SFTデータ量・応答形式・Tokenizer/context lengthがボトルネックであり、基盤Token数を増やすだけでは不十分と判断します。SFTは一般文・会話・医療の基盤を失わないよう、rehearsal ratio 0.20を固定します。

## 使用するデータと再現条件

設定は[`configs/issue1-both-50m-sft-from-5m-rehearsal020-3k.toml`](../../configs/issue1-both-50m-sft-from-5m-rehearsal020-3k.toml)です。モデル構造は075と同じvocab 4,096、dim 576、12層、9 heads、context length 256、RoPE、LayerNorm、SwiGLU、実測50,207,616 parametersです。初期checkpointは`artifacts/checkpoints/issue1-both-50m-pretrain-5m-2p5k/best.pt`で、SHA-256は`71931b2c689c2fbaa31c8c92c022a21fac571894ec2993a59be48644794e5e17`です。

SFT trainは`artifacts/sft/issue1-both-balanced-v1/train.npz`、SHA-256 `645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`、validationは`artifacts/sft/issue1-both-full-v1/validation.npz`、SHA-256 `fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`です。rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。Tokenizerは075と同じ`mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

学習条件はbatch size 8、最大3,000 step、evaluation・生成間隔100、checkpoint間隔500、AdamW、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 42、response-only loss、EOS loss weight 0.50、rehearsal ratio 0.20です。SFT train batchとrehearsal batchの損失を分けて保存します。生成サンプルは`<|startofconversation|><|speaker:DA|>こんにちは！<eos:3><|speaker:DC|>`形式で保存します。

予定コマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-50m-sft-from-5m-rehearsal020-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-5m-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-50m-sft-from-5m-rehearsal020-3k \
  --samples-dir artifacts/samples/issue1-both-50m-sft-from-5m-rehearsal020-3k \
  --lr-schedule-steps 3000 \
  --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 \
  --sample-template conversation \
  --sample-speaker-a DA \
  --sample-speaker-b DC \
  --device mps
```

Colab CLIでT4割当を先に試し、bundleには075 best checkpoint、加工済みSFTデータ、rehearsal Token列、Tokenizer、コード、設定だけを含めます。元JSONL、医師国家試験原本、`medilink_analysis`は含めません。503や割当上限で失敗した場合は、エラーとsession状態をノートへ残して同一条件のMPSへ切り替えます。

開始commitは`6b8adf4`で、GitHubの`origin/main`へpush済みです。設定ファイルのSHA-256は`42b378bef6c16cdd6335acf92fc6e6ab2afd5e17e24db0e63108d2b782c32a27`、bootstrapのSHA-256は`eb3473ec0428f51afda2d0bfdd40572b78dc1105c6bd3c9d9889c9ec586a67c9`です。Colab送信用bundleは`/tmp/small_llm-colab-076-XXXXXX.tar.gz`、236,430,874 bytes、SHA-256 `622c5e5586409f4c726abc6b274e1f3999737061bf363c220528be9f7586636b`です。bundleには075 best重み、入力hash検証に必要なmetadata、SFTデータ、rehearsal Token列、Tokenizer、コードと設定だけを含め、原JSONL・医師国家試験原本・`medilink_analysis`は含めていません。

## 成功・失敗の判定基準

3,000 stepをNaN、OOM、shape error、base checkpoint signature不一致なく完走し、100 step間隔のlossと生成文、500 step間隔のcheckpoint metadata、summaryを保存できれば実装上の成功とします。品質比較では074と同じ5領域、固定chat-test-v1、EOS到達率、平均生成長、short・medium・long別F1を記録します。lossやToken overlapだけで自然さ、医学的正確性、会話能力を断定しません。失敗や崩れた生成も削除せず保存します。

## 実験中の記録

この節にはColab試行、bundleのhash、MPS切り替え、開始時のparameter数、100 stepごとのSFT・rehearsal・validation loss、生成本文、500 stepごとのmetadata、警告、途中停止を追記します。075のbest checkpointとデータhashを上書きせず、実行backendが変わった場合も時系列で残します。

2026年9月6日、MPS学習前に`colab sessions`を実行し、`No active sessions found on server.`を確認しました。その後、`colab new --session exp076-both-50m-sft-from-5m --gpu T4`を実行しましたが、assignment endpointがHTTP 503 `Service Unavailable`を返して終了しました。Colab側のbundle upload、入力hash検証、SFT初期化、学習stepは発生していません。この失敗を成功実験と混ぜず、同一条件のMPSへ切り替えます。

同日、MPSで076 SFTを開始しました。step 1のtrain lossは4.183418、SFT train lossは4.239130、rehearsal train lossは3.960572、validation lossは4.342213、PPL 76.88、learning rateは5.0e-7、経過4.12秒でした。実測parameter数は075と同じ50,207,616で、SFT設定はrehearsal ratio 0.20、EOS loss weight 0.50、lr schedule終点3,000 stepです。NaN、OOM、shape error、base checkpoint signature不一致はありません。学習を継続します。

step 100はtrain loss 3.678967、SFT train loss 3.792676、rehearsal loss 3.224131、validation loss 3.811520、PPL 45.22、経過65.23秒でした。step 200は3.999521、4.261581、2.951283、3.844824、PPL 46.75、経過218.40秒、step 300は3.590491、3.882473、2.422562、3.795923、PPL 44.52、経過338.52秒、step 400は3.726098、3.695177、3.849782、3.829993、PPL 46.06、経過464.20秒でした。step 500ではtrain loss 4.052101、SFT train loss 4.131844、rehearsal loss 3.733128、validation loss 3.828892、PPL 46.01、learning rate 4.7931e-5、経過590.58秒となりました。step 500 metadataのweights SHA-256は`3cbb4f447700da2c7bcd31c37c8c15cf66aee1d47cff1961d47027b8c96275e9`です。固定会話promptの生成はstep 0の「こんばんは!」から、step 500では「こんにちは!!こんばんは!」へ変化し、挨拶形式への適応は見えますが、まだ過剰な反復があります。validation lossはstep 100以降おおむね3.8前後で横ばいです。NaN、OOM、shape error、警告はありません。学習を継続します。

step 600はtrain loss 3.535798、SFT train loss 3.595123、rehearsal loss 3.298497、validation loss 3.777404、PPL 43.70、経過721.49秒、step 700は4.251809、4.130155、4.738423、3.745436、PPL 42.33、経過849.32秒、step 800は3.636042、3.590926、3.816505、3.754749、PPL 42.72、経過977.32秒、step 900は3.881740、3.854248、3.991708、3.728051、PPL 41.60、経過1103.05秒でした。step 1,000ではtrain loss 3.266225、SFT train loss 3.184397、rehearsal loss 3.593535、validation loss 3.694405、PPL 40.22、learning rate 4.0147e-5、経過1230.73秒となりました。step 500からvalidation lossは0.134487低下し、step 1,000 metadataのweights SHA-256は`a3c9454da9e82550705430ffff33000dbf4a381d44ef26b0f3745ad3ea417c8b`です。固定会話promptはstep 1,000で「こんにちはー!」へ寄り、step 500より反復が減りましたが、短い挨拶だけで会話内容の適合性はまだ判断できません。NaN、OOM、shape error、警告はありません。学習を継続します。

step 1,100はtrain loss 3.485310、SFT train loss 3.597784、rehearsal loss 3.035416、validation loss 3.693570、PPL 40.19、経過1365.07秒、step 1,200は3.424699、3.588538、2.769341、3.682499、PPL 39.75、経過1493.67秒、step 1,300は4.067103、4.147092、3.747147、3.680499、PPL 39.67、経過1620.36秒、step 1,400は4.173775、4.295277、3.687765、3.665573、PPL 39.08、経過1743.04秒でした。step 1,500ではtrain loss 3.612076、SFT train loss 3.904130、rehearsal loss 2.443856、validation loss 3.665683、PPL 39.08、learning rate 2.8742e-5、経過1868.35秒となりました。step 1,000からvalidation lossは0.028722低下し、step 1,500 metadataのweights SHA-256は`d5924ac9acd8f2c2e157926e3c0f75c3a37ba9c9538ce58597015bf0d1672be0`です。validationはstep 1,400の3.665573が暫定最良です。固定会話promptはまだ短い挨拶中心で、会話内容の改善はこの固定promptだけでは判断できません。NaN、OOM、shape error、警告はありません。学習を継続します。

step 1,600はtrain loss 3.452546、SFT train loss 3.383689、rehearsal loss 3.727973、validation loss 3.639713、PPL 38.08、経過1993.18秒、step 1,700は3.331576、3.273077、3.565571、3.623976、PPL 37.49、経過2117.97秒、step 1,800は3.620782、3.701358、3.298477、3.608459、PPL 36.91、経過2245.13秒、step 1,900は3.070168、3.164855、2.691417、3.594561、PPL 36.40、経過2370.07秒でした。step 2,000ではtrain loss 4.154740、SFT train loss 4.238073、rehearsal loss 3.821408、validation loss 3.578288、PPL 35.81、learning rate 1.6982e-5、経過2493.58秒となりました。step 1,500からvalidation lossは0.087395低下し、step 2,000 metadataのweights SHA-256は`06748d3dfa4677ee2599ded2a17e7f6c83795128e601b736bec39953d52d9155`です。固定会話promptは短い挨拶中心であり、validation改善だけから会話品質を断定しません。ここまでNaN、OOM、shape error、警告はありません。学習を継続します。

step 2,100はtrain loss 3.326718、SFT train loss 3.425356、rehearsal loss 2.932166、validation loss 3.568645、PPL 35.47、経過2630.31秒、step 2,200は3.041746、2.850117、3.808261、3.558014、PPL 35.09、経過2759.15秒、step 2,300は3.331714、3.192138、3.890019、3.555646、PPL 35.01、経過2884.73秒、step 2,400は3.486757、3.381741、3.906820、3.539297、PPL 34.44、経過3013.98秒でした。step 2,500ではtrain loss 3.206680、SFT train loss 3.026942、rehearsal loss 3.925631、validation loss 3.536901、PPL 34.36、learning rate 8.2333e-6、経過3147.24秒となりました。step 2,000からvalidation lossは0.041386低下し、step 2,500 metadataのweights SHA-256は`165525cfaae4660b3df91fbef86dff144e397b39b283390c0e12a0fc89fff01d`です。固定会話promptは短い挨拶中心で、validation改善だけから会話品質を断定しません。ここまでNaN、OOM、shape error、警告はありません。学習を継続します。

step 600はtrain loss 3.535798、SFT train loss 3.595123、rehearsal loss 3.298497、validation loss 3.777404、PPL 43.70、経過721.49秒、step 700は4.251809、4.130155、4.738423、3.745436、PPL 42.33、経過849.32秒、step 800は3.636042、3.590926、3.816505、3.754749、PPL 42.72、経過977.32秒、step 900は3.881740、3.854248、3.991708、3.728051、PPL 41.60、経過1103.05秒でした。step 1,000ではtrain loss 3.266225、SFT train loss 3.184397、rehearsal loss 3.593535、validation loss 3.694405、PPL 40.22、learning rate 4.0147e-5、経過1230.73秒となりました。step 500からvalidation lossは0.134487低下し、step 1,000 metadataのweights SHA-256は`a3c9454da9e82550705430ffff33000dbf4a381d44ef26b0f3745ad3ea417c8b`です。固定会話promptはstep 1,000で「こんにちはー!」へ寄り、step 500より反復が減りましたが、短い挨拶だけで会話内容の適合性はまだ判断できません。NaN、OOM、shape error、警告はありません。学習を継続します。

## 実験終了後の結果と解釈

完走後に実際のruntime、最良・最終loss、SFTとrehearsalの損失、学習時間、best checkpoint hash、5領域評価、固定chat評価、生成本文を追記します。074との差は、初期checkpointとSFT条件を分けて記載し、075基盤の効果とSFT条件の効果を混同しません。

## 次に試すこと

076で075基盤のSFT効果を確認した後、会話データの長い応答比率を変更するか、現代的なinstruction形式と通常pretrainingの混合比率を一つだけ変更します。SFT後も長文会話が改善しない場合は、context length 512への拡張を別実験として検証します。
