# 実験071：068の長文層化条件をseed 123で再確認

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。GitHubの[Issue #1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)を今後の実験候補として確認しました。Issueの中心は、青空文庫などの標準的な文章だけでは得にくい、現代の雑談における短文応答、相づち、砕けた語尾、話者交代、複数人会話、話題継続を測定することです。既存の実験ではRPCとMRMPを通常の日本語データへ組み込み、医師国家試験由来のデータも一般モデルの保持領域として扱っています。医療専用モデルへは寄せず、元の`/Users/koseki/projects/medilink_analysis`とその原データは変更・削除しません。

実験067〜070では、20MモデルへRPC・MRMPの会話SFTと一般日本語rehearsalを組み合わせ、SFT部分の長文例の割合を0/6、1/6、2/6、3/6と比較しました。2/6に相当する実験068は固定chat-testの全体F1が0.220352、long F1が0.162121で、3/6の実験070より全体・short・medical保持のバランスが良好でした。ただし、各条件はseed 42の一回だけであり、2/6の改善が学習条件そのものによるものか、抽出されたミニバッチと最適化経路の偶然によるものかは確認できていません。

今回は実験068とデータ、モデル、学習率、step数、評価方法を固定し、学習seedだけを123へ変更します。長文はSFT部分6例のうち`round(6 * 0.25)=2`例、応答24 Token以上と定義します。評価用chat-testのseedは42に固定し、学習seedの差と評価サンプリングの差を混同しないようにします。Issue #1の新規コーパスをさらに追加する実験ではありません。まず既存の会話混合条件の再現性を確かめ、データ追加とsampling設計を分離した上で次の実験へ進みます。

仮説は、seed 123でも2/6条件が0/6の実験067よりlong F1を改善し、3/6の実験070ほどshort F1・medical loss・precisionを損なわないことです。単一seedの068を完全に再現する必要はありませんが、全体F1、long F1、short F1、5領域lossの大きな順位逆転がなければ、2/6を次の人手レビューおよび50M拡張の暫定条件とします。逆に順位が崩れる場合は、長文割合の結論を保留し、追加seedまたはミニバッチ抽出の分散を調べます。Token overlapは自然さそのものではないため、生成本文の話題逸脱、反復、話者marker維持も記録します。

## 再現条件

実験070の完了後、origin/mainにpush済みの基準commitは`1c1d456`です。本実験で追加する設定ファイルは[`configs/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed123.toml`](../../configs/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed123.toml)です。モデルはRoPE・LayerNorm・SwiGLU、dim 384、10層、6 heads、context length 256、19,308,032 parametersです。Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`です。

設定ファイルのSHA-256は`132daeab437f28fe2d2f119bcaf12efba06e9c541f1cf97bbef5966c8a485612`です。入力ファイルのSHA-256は、base checkpointが`326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`、Tokenizerが`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`、SFT trainが`645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`、SFT validationが`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`、rehearsal Token列が`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。

入力は実験068と同じbase checkpoint、SFT train・validation、rehearsal Token列を使います。base checkpointは`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`、SFT trainは`artifacts/sft/issue1-both-balanced-v1/train.npz`、SFT validationは`artifacts/sft/issue1-both-full-v1/validation.npz`、rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`です。SFT trainは64,423例・response 770,990 Tokenで、応答24 Token以上の長文は4,286例です。validationは49,045例・response 738,660 Tokenで、同じ長さの長文は6,890例です。

学習条件はbatch size 8、3,000 step、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 123、EOS loss weight 0.50、rehearsal ratio 0.20です。SFTとrehearsalを0.80対0.20で合算し、SFT部分6例から長文2例・通常4例を抽出します。MPSではAMPを使いません。生成はconversation形式、話者DAとDC、固定promptは`こんにちは！`、最大160 Token、temperature 0.8、top-k 40です。

再現に使うコマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed123.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed123 \
  --samples-dir artifacts/samples/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed123 \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --long-response-ratio 0.25 \
  --long-response-min-tokens 24 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

学習開始前に設定ファイルおよび入力ファイルのSHA-256を取得し、学習中の500 stepごとにmetricsと生成本文を保存します。Colab CLIでT4割り当てを先に試し、失敗した場合はHTTP応答とsession状態をこのノートへ記録してMPSへ切り替えます。

## 成功・失敗の判定基準

3,000 stepをNaN、OOM、shape errorなく完走し、step 0〜3,000の生成本文、metrics、summary、checkpoint metadata、5領域評価、固定chat-test 48例を保存できれば実装上の成功とします。性能面では実験068との単純な一致ではなく、067・068・070の範囲でseed 123の位置を確認します。生成が悪化していても削除せず、結果をそのままGitHubへ保存します。原因が分からない場合は原因不明と明記し、次に調べる内容を残します。

## 実験中の記録

この節には、Colab試行、MPS切り替え、500 stepごとのtrain loss・validation loss・PPL・経過時間・学習率・生成本文、警告や途中停止を時系列で追記します。学習中の出力は省略せず保存します。

2026年9月6日、MPS学習の開始前に`colab new -s exp071-both-long025-seed123 --gpu T4`を実行しました。しかしColab CLIのassignment endpointがHTTP 503 `Service Unavailable`を返し、セッション作成に失敗しました。直後の`colab sessions`は`No active sessions found on server.`でした。bundle uploadやColab上の学習は発生していないため、同一条件をMPSで実行します。

同日、Colab失敗を記録したcommit `1e62bdb`の後、MPSで学習を開始しました。step 1はtrain loss 4.342813、SFT loss 4.578291、rehearsal loss 3.400899、validation loss 4.723621、PPL 112.5752、経過時間2.34秒でした。step 100はvalidation loss 4.109018、PPL 60.8869、step 200は4.040311、PPL 56.8440、step 300は4.008517、PPL 55.0651、step 400は3.991986、PPL 54.1623でした。step 500ではtrain loss 4.036089、SFT loss 4.275889、rehearsal loss 3.076891、validation loss 3.935468、PPL 51.1861、learning rate 4.7931e-5、経過時間214.19秒となりました。step 0〜500のmetrics、checkpoint metadata、生成本文を保存しました。step 500の固定prompt生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんにちは。`で、EOSへ到達しています。ここまでNaN、OOM、shape error、警告はありません。学習は継続中です。

step 600ではvalidation loss 3.915794、PPL 50.1889、step 700では3.908716、PPL 49.8349、step 800では3.894884、PPL 49.1504、step 900では3.869629、PPL 47.9246となりました。step 1,000ではtrain loss 4.113091、SFT loss 4.459245、rehearsal loss 2.728475、validation loss 3.832691、PPL 46.1867、learning rate 4.0147e-5、経過時間472.66秒となりました。step 600〜1,000の生成本文、step 1,000のcheckpoint metadata、metricsを保存しました。step 1,000の固定prompt生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんにちは!`で、EOSへ到達しています。step 800付近にvalidationの改善が少し緩む揺らぎはありましたが、step 900〜1,000では改善し、ここまで異常はありません。学習は継続中です。

step 1,100ではvalidation loss 3.833365、PPL 46.2178、step 1,200では3.826299、PPL 45.8924、step 1,300では3.806826、PPL 45.0073、step 1,400では3.807464、PPL 45.0361となりました。step 1,500ではtrain loss 3.757710、SFT loss 3.663838、rehearsal loss 4.133196、validation loss 3.796484、PPL 44.5443、learning rate 2.8742e-5、経過時間738.59秒となりました。step 1,100〜1,500の生成本文、step 1,500のcheckpoint metadata、metricsを保存しました。step 1,500の固定prompt生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> そうです!`で、EOSへ到達しています。validationはstep 1,400でほぼ横ばいになった後、step 1,500で改善しました。ここまで異常はありません。学習は継続中です。

step 1,600ではvalidation loss 3.788731、PPL 44.2003、step 1,700では3.763321、PPL 43.0913、step 1,800では3.745196、PPL 42.3173、step 1,900では3.750456、PPL 42.5405となりました。step 2,000ではtrain loss 3.648080、SFT loss 3.543768、rehearsal loss 4.065326、validation loss 3.732562、PPL 41.7860、learning rate 1.6982e-5、経過時間1007.76秒となりました。step 1,600〜2,000の生成本文、step 2,000のcheckpoint metadata、metricsを保存しました。step 2,000の固定prompt生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんばんはー。`で、EOSへ到達しています。step 1,900で小さな反発がありましたが、step 2,000では再び改善しています。ここまで異常はありません。学習は継続中です。

step 2,100ではvalidation loss 3.739756、PPL 42.0877、step 2,200では3.747287、PPL 42.4059、step 2,300では3.738759、PPL 42.0458、step 2,400では3.737458、PPL 41.9911となりました。step 2,500ではtrain loss 3.577265、SFT loss 3.119239、rehearsal loss 5.409369、validation loss 3.729172、PPL 41.6446、learning rate 8.2333e-6、経過時間1278.62秒となりました。step 2,100〜2,500の生成本文、step 2,500のcheckpoint metadata、metricsを保存しました。step 2,500の固定prompt生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんばんはー!`で、EOSへ到達しています。rehearsal train lossがstep 2,500で高くなりましたが、validation lossは今回の実行中で最良です。ここまで異常はありません。学習は継続中です。

step 2,600ではvalidation loss 3.724659、PPL 41.4571、step 2,700では3.723881、PPL 41.4249、step 2,800では3.726536、PPL 41.5350、step 2,900では3.713784、PPL 41.0087となりました。最終step 3,000ではtrain loss 3.268579、SFT loss 3.373382、rehearsal loss 2.849367、validation loss 3.713886、PPL 41.0129、learning rate 5.0000e-6、経過時間1550.76秒となりました。step 2,600〜3,000の生成本文、step 3,000のcheckpoint metadata、metrics、summaryを保存しました。最良checkpointはstep 2,900で、学習はNaN、OOM、shape errorなく完走しました。step 3,000の固定prompt生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんにちは!`で、EOSへ到達しています。評価用のcheckpointは最良の`best.pt`を使用します。

学習完了後の評価はCPU上で行います。5領域については`evaluate_torch.py domains`へ`general=artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、`conversation=artifacts/tokens/mixed-ja-80-10-10-v2-conversation-val.bin`、`medical=artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin`、`RPC=artifacts/tokens/issue1-real-persona-chat-validation.bin`、`MRMP=artifacts/tokens/issue1-mrmp-validation.bin`を渡し、20バッチずつ測定します。固定chat-testは`artifacts/corpus/conversation-v1/test.jsonl`と`experiments/evaluation/chat-test-v1.json`を使い、48例、最大160 Token、temperature 0.8、top-k 40、評価seed 42で実行します。出力JSONと全文TXTを保存してから、067・068・070との比較を完成させます。

## 実験終了後の結果と解釈

学習終了直後に、実際のbackend、最良checkpoint、学習時間、5領域loss、固定chat-testのEOS・長さ・precision・recall・F1、stratum別およびsource別集計、生成本文の質的観察を追記します。068との差分だけでなく、067と070を含む比較表を残し、2/6条件を採用・保留・棄却のいずれかに判断します。

学習はMPS上で完走し、PyTorch 2.14.0、AMP無効、19,308,032 parametersでした。summary上の経過時間は1551.04秒、最良checkpointはstep 2900、`best.pt`のSHA-256は`4e42142faa610560e7fad86835cffeabddfe1b1d589ff83df387d9e0f69465dc`です。最良validation lossは3.713784（PPL 41.0056相当）で、最終step 3000はvalidation loss 3.713886、PPL 41.0129でした。step 0〜3,000の全31個の生成本文と500 stepごとのcheckpoint metadataを保存し、NaN、OOM、shape errorは発生していません。

5領域評価はCPU上で20バッチずつ行いました。generalはvalidation loss 5.473651（PPL 238.3287）、conversationは2.894462（18.0738）、medicalは3.214831（24.8991）、RPCは2.879695（17.8088）、MRMPは2.329470（10.2725）でした。実験068との差はgeneral +0.012179、conversation +0.010222、medical -0.017538、RPC +0.006786、MRMP +0.001559です。一般・会話・RPC・MRMPのlossは068より悪化しましたが、medical lossは改善しました。実験067との差ではgeneral +0.003770、conversation -0.003127、medical -0.003402、RPC -0.020235、MRMP -0.011943で、領域ごとに挙動が分かれています。

固定chat-test v1は48例すべてでEOSへ到達しました。平均生成Token数は13.4792、precisionは0.243643、recallは0.204415、全体Token overlap F1は0.177428でした。short・medium・long別F1はそれぞれ0.263437、0.137256、0.131591です。実験068との差は、平均生成長+2.2708、precision-0.054153、recall-0.025861、全体F1-0.042924、short F1-0.053683、medium F1-0.044558、long F1-0.030530です。実験067との差は、平均生成長+3.0000、全体F1-0.030325、short F1-0.059232、medium F1-0.029587、long F1-0.002158です。長く生成できてもToken overlapは改善せず、seed 42の068で見えたlong層の改善はseed 123では再現しませんでした。

source別では、MRMP 24例のF1が0.182623、平均生成Token数が10.2917、RPC 24例のF1が0.172233、平均生成Token数が16.6667でした。両sourceとも24/24でEOSへ到達しています。長い会話では、直前の話題に接続せず別の料理・スポーツ・道具の話へ移る出力や、短い相づちだけで終わる出力が残っています。したがって、今回の数値だけから2/6条件を採用することはできず、068の改善は「再現性未確認」と判断します。人手レビュー用の48例テンプレートも生成しましたが、`review_status`は`pending_human_review`のままであり、まだ人手判定を完了したとは扱いません。

成果物のSHA-256は、metricsが`f13bb25af8ac2589970563b7e9c52b2765144dc361787a9861f40182ffc68b7b`、summaryが`8cfb204f9dbfad61437f6bfb9de206e2a5f006a026de5fc81c7c0ab299cc664b`、best metadataが`d9d9e959438510620adeda6a561cb481a4beb8f67d558d996af60b348fdc4dbe`、5領域評価JSONが`f5b6ef65120d77744e0383865b15d982fbdc0dcc54d44d796ed9717efd6a816b`、固定chat評価JSONが`5a45f4d1455baa27c18041612db56ad5f9ee883e94f286725a3607d623dcbbf0`、固定chat全文TXTが`b10db12bf957293b8ec3a3fff929c683be60ad39ec639e949ab9381072e95f5e`、人手レビュー用JSONが`20d4f5c77eb3dd39f08517de507350ca5eb91dcecffa800bcbb422b23d5a7281`です。

評価結果は[checkpoint metadata](../../artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed123/)、[学習中の生成サンプル](../../artifacts/samples/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed123/)、[5領域評価JSON](../../artifacts/evaluations/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed123-domains.json)、[固定chat評価JSON](../../artifacts/evaluations/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed123-chat-test-v1.json)、[固定chat全文TXT](../../artifacts/evaluations/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed123-chat-test-v1.txt)、[人手レビュー用テンプレート](../../artifacts/evaluations/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-seed123-chat-review.json)から確認できます。重い`.pt`本体はGit管理外ですが、metadataにSHA-256を保存しています。

## 次に試すこと

seed 123で068のlong F1改善が再現しなかったため、seed 777を同一条件で追加し、3 seed（42・123・777）の平均と分散を確認します。seed間の分散が大きければ、人手レビューと評価セットの拡張を先に行い、長文割合の採用を保留します。再現性が確認できた場合に限り、20Mで選んだ条件を50Mへ拡大します。
