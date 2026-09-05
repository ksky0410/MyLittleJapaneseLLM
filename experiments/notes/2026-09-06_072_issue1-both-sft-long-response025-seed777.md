# 実験072：068の長文層化条件をseed 777で再確認

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。GitHubの[Issue #1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)を、現代的な日本語会話を標準日本語と比較するための継続的な実験候補として扱います。Issueが求めているのは、RPC・MRMPに含まれる相づち、砕けた語尾、短文応答、話者交代、複数人会話、話題継続の確認です。一般日本語、会話データ、医師国家試験由来データを同じモデルの目的別評価へ使い、医療専用モデルにはしません。元の`/Users/koseki/projects/medilink_analysis`と原データは保全し、変更・削除しません。

実験067〜070では、20MモデルにRPC・MRMPの会話SFTと一般日本語rehearsalを組み合わせ、SFT部分6例に含める応答24 Token以上の長文数を0、1、2、3と比較しました。2例条件の実験068（seed 42）は全体F1 0.220352、long F1 0.162121でしたが、同じ条件をseed 123で再実行した実験071は全体F1 0.177428、long F1 0.131591でした。071ではmedical lossは改善したものの、068で見えた会話F1の改善は再現していません。この差は長文層化効果がないこと、または20M・3,000 step・48例評価における乱数分散が大きいことのどちらでも説明できます。

今回は実験068・071とデータ、モデル、sampling、学習率、step数、評価方法を固定し、学習seedだけを777へ変更します。rehearsal ratioは0.20、SFT部分は6例、長文は2例（応答24 Token以上）です。評価seedは42に固定し、3条件の評価サンプリングを共通化します。新しいコーパスの追加やモデル構造変更を同時に行わず、seed分散だけを測定します。

仮説は、seed 777が068と071の間に入り、3 seedの結果から2/6条件の平均性能と分散を計算できることです。777でも068と同様にF1・long F1が高ければ長文層化の有効性を暫定採用します。777が071に近ければ、単一seedで見えた改善は不安定と判断し、2/6条件を確定せず、人手レビューと評価セット拡張を先に行います。いずれの場合も、EOS、生成長、precision、recall、長さ別F1、source別値、5領域loss、生成本文を保存します。

## 再現条件

実験071の評価まで完了し、origin/mainへpush済みの基準commitは`872b169`です。本実験の設定ファイルは[`configs/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed777.toml`](../../configs/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed777.toml)です。モデルはRoPE・LayerNorm・SwiGLU、dim 384、10層、6 heads、context length 256、19,308,032 parametersです。Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`です。

設定ファイルのSHA-256は`d5173226eaca440a302d51d67cef194f53a1d585f4ca3a16fcba936c6e6630e4`です。入力ファイルのSHA-256は、base checkpointが`326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`、Tokenizerが`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`、SFT trainが`645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`、SFT validationが`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`、rehearsal Token列が`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。

入力は実験068・071と同じbase checkpoint、SFT train・validation、rehearsal Token列を使います。base checkpointは`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`、SFT trainは`artifacts/sft/issue1-both-balanced-v1/train.npz`、SFT validationは`artifacts/sft/issue1-both-full-v1/validation.npz`、rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`です。SFT trainは64,423例・response 770,990 Tokenで、応答24 Token以上は4,286例です。validationは49,045例・response 738,660 Tokenで、同じ長さの例は6,890例です。

学習条件はbatch size 8、3,000 step、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 777、EOS loss weight 0.50、rehearsal ratio 0.20です。SFTとrehearsalを0.80対0.20で合算し、SFT部分6例から長文2例・通常4例を抽出します。MPSではAMPを使いません。生成はconversation形式、話者DAとDC、固定promptは`こんにちは！`、最大160 Token、temperature 0.8、top-k 40です。

再現に使うコマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed777.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed777 \
  --samples-dir artifacts/samples/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed777 \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --long-response-ratio 0.25 \
  --long-response-min-tokens 24 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

設定ファイルのSHA-256と、実際に読み込む入力のSHA-256を開始前に記録します。学習中はstep 500、1000、1500、2000、2500、3000で生成本文、metrics、checkpoint metadataを保存し、500 stepごとにcommit・pushします。Colab CLIでT4割り当てを試し、失敗時は応答とsession状態を記録してMPSへ切り替えます。

## 成功・失敗の判定基準

3,000 stepをNaN、OOM、shape errorなく完走し、step 0〜3,000の生成本文、metrics、summary、checkpoint metadata、5領域評価、固定chat-test 48例、人手レビュー用テンプレートを保存できれば実装上の成功とします。研究上は、seed 42・123・777の三点比較に使える同一形式の成果物が揃うことを成功条件とします。性能の良し悪しは別途、平均と分散、生成本文、人手レビューで判定します。失敗や悪い生成は削除せず、そのまま記録します。

## 実験中の記録

この節には、Colab試行、MPS切り替え、500 stepごとのtrain loss・validation loss・PPL・経過時間・学習率・生成本文、警告や途中停止を時系列で追記します。学習中の出力は省略せず保存します。

2026年9月6日、MPS学習の開始前に`colab new -s exp072-both-long025-seed777 --gpu T4`を実行しました。しかしColab CLIのassignment endpointがHTTP 503 `Service Unavailable`を返し、セッション作成に失敗しました。直後の`colab sessions`は`No active sessions found on server.`でした。bundle uploadやColab上の学習は発生していないため、同一条件をMPSで実行します。

同日、Colab失敗を記録したcommit `13be73e`の後、MPSで学習を開始しました。step 1はtrain loss 4.789038、SFT loss 4.726177、rehearsal loss 5.040481、validation loss 4.723932、PPL 112.6102、経過時間2.27秒でした。step 100はvalidation loss 4.099635、PPL 60.3182、step 200は4.043266、PPL 57.0122、step 300は4.002629、PPL 54.7419、step 400は3.981801、PPL 53.6135でした。step 500ではtrain loss 4.149330、SFT loss 4.370154、rehearsal loss 3.266031、validation loss 3.959216、PPL 52.4162、learning rate 4.7931e-5、経過時間222.12秒となりました。step 0〜500のmetrics、checkpoint metadata、生成本文を保存しました。step 500の固定prompt生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんにちはよ!`で、EOSへ到達しています。ここまでNaN、OOM、shape error、警告はありません。学習は継続中です。

step 600ではvalidation loss 3.929026、PPL 50.8574、step 700では3.920575、PPL 50.4294、step 800では3.902041、PPL 49.5034、step 900では3.879163、PPL 48.3837となりました。step 1,000ではtrain loss 3.661442、SFT loss 3.639277、rehearsal loss 3.750103、validation loss 3.883222、PPL 48.5805、learning rate 4.0147e-5、経過時間484.01秒となりました。step 600〜1,000の生成本文、step 1,000のcheckpoint metadata、metricsを保存しました。step 1,000の固定prompt生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんばんはー`で、EOSへ到達しています。step 900から1,000でvalidationが小さく反発しましたが、ここまで異常はありません。学習は継続中です。

## 実験終了後の結果と解釈

学習終了直後に、実際のbackend、最良checkpoint、学習時間、5領域loss、固定chat-testのEOS・長さ・precision・recall・F1、stratum別およびsource別集計、生成本文の質的観察、071・068との差分を追記します。三つの学習seedをまとめ、2/6条件の採用・保留・棄却を判断します。

## 次に試すこと

三つのseedで再現性が確認できれば、Issue #1の会話データを含む20M条件を50Mへ拡張します。再現性が弱ければ、長文割合の最適化を続ける前に人手レビューの入力を整え、話題適合・役割適合・崩壊を評価します。その後、必要に応じてSFTとrehearsalのToken予算を独立制御する実装を検討します。
