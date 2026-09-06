# 実験080：50M日本語モデルの10M Token事前学習

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。実験079で長文応答のoversamplingは全体の会話性能を改善しなかったため、今回はSFTの細かな比率ではなく、事前学習のデータ量を増やすことを試します。強い教師モデルによる蒸留は行わず、同じ50Mモデルを日本語コーパスだけから学習します。

実験075では、同じ50M構造を約5M Token、2,500 stepで事前学習しました。実験080では、同じTokenizer・モデル構造・seed・optimizerを保ち、FineWeb2 Edu Japanese、青空文庫、Wikipedia、日本語会話、医師国家試験由来データを混ぜた約10M Token列を使い、5,000 stepまで学習します。batch size 8、context length 256なので、学習で見るToken数はおよそ10.24Mです。075の約5.12M Tokenに対して、学習予算をほぼ2倍にする比較です。

仮説は、学習Token数を約2倍にすると、generalだけでなくconversation、medical、RPC、MRMPのvalidation lossが改善し、固定chat-testの応答がより長く自然になることです。ただし、10M列ではWikipediaの比率が増え、075の5M列とsource比率が完全には一致しません。したがって結果を「Token数だけの因果効果」と断定せず、データ量増加とsource構成変更を合わせた実用的な主線候補として扱います。改善しなければ、次は同一5M列を複数周回した条件を別実験にして、データの多様性と反復学習を分離します。

成功条件は、5,000 stepを完走し、100 step間隔のmetricsと生成文、500 step間隔のcheckpoint metadata、最良checkpoint、summary、5領域評価、固定chat-test 48例を保存することです。validation lossだけでなく、EOS到達率、平均生成長、全体および長さ別のToken overlap F1、生成文の日本語としての自然さを確認します。

## 再現条件

モデルはRoPE・LayerNorm・SwiGLU、dim 576、12層、9 heads、context length 256、MLP倍率4、50M級の構造です。TokenizerはSentencePiece Unigram、vocab 4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`です。学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1-train.bin`、9,999,973 Token、SHA-256は`d043d06180d2c6deb0e0c14038fd1b3f736f86f062cf61260bd19282f8ce48e4`です。general validationは`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。元の`medilink_analysis`と医師国家試験原データは変更せず、small_llm側の加工済みToken列だけを読み取ります。

10Mコーパスのmanifestは`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1.manifest.json`です。Token比率はaozora 5.08%、fineweb 42.18%、wikipedia 42.19%、conversation 5.27%、medical 5.27%です。075の5M列はaozora 10.17%、fineweb 71.87%、conversation 8.98%、medical 8.98%だったため、この差を結果の解釈に明記します。

設定ファイルは`configs/issue1-both-50m-pretrain-10m-5k.toml`です。学習条件はbatch size 8、最大5,000 step、eval/sample interval 100、checkpoint interval 500、eval batches 20、learning rate 3e-4から3e-5、warmup 500、weight decay 0.1、seed 42です。設定ファイルのSHA-256は学習開始前に計算して追記します。

開始前に確認したSHA-256は、設定ファイルが`1f3570dd38e13286e9dc3270e68f2b5f803dd8aa44410f2ce85abbede56b9447`、10M corpus manifestが`f9d17b36998671320ab69d7448fde10a7be0c2ba894ae0daa00fc437ac3e2c64`、Tokenizerが`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。新規スクリプトは`py_compile`を通過しました。

ローカルでの再現コマンドは次のとおりです。

```bash
uv run python scripts/train_torch.py \
  --config configs/issue1-both-50m-pretrain-10m-5k.toml \
  --device mps
```

Colab T4が利用できる場合は、同じ設定と入力を`colab_bootstrap_080.py`から実行します。Colab失敗時はHTTP status、session状態、bundle hashを残してMPSへ切り替えます。

開始時点で`colab sessions`は`No active sessions found on server.`でした。Colab送信用bundleは`/tmp/small_llm-colab-080.tar.gz`、12MB、SHA-256は`ec8d498e1956083df20333b094aba73b333fb32657c04987f9a7f7a8a51552c5`です。bundleには080の設定、学習スクリプト、srcパッケージ、Tokenizer、10M Token列、general validationだけを含め、元JSONL、医師国家試験原本、`medilink_analysis`は含めていません。

## 実験中の記録

学習開始前に設定・入力hash・Git commit・bundle hash・Colab試行結果を追記します。学習中は1,000 stepを超えて記録を空けず、原則100 stepごとにmetrics、生成文、異常を保存します。生成文は良いものだけでなく、崩れた出力も削除せずGitHubへ追加します。

Colab T4の新規session `exp080-both-50m-pretrain-10m`は作成に成功し、bundle uploadと入力hash検証も完了しました。Colab側のTorch/CUDA情報は学習完了後のsummaryから確定します。標準出力の転送は遅れましたが、リモートの`metrics.jsonl`を途中回収して学習継続を確認しています。step 1はtrain loss 8.790941、validation loss 8.819555、PPL 6765.26、learning rate 6.0e-7、経過3.48秒でした。step 100はvalidation loss 7.276060、PPL 1445.28、learning rate 6.0e-5、経過12.51秒、step 500はtrain loss 5.497313、validation loss 6.379793、PPL 589.81、learning rate 3.0e-4、経過56.60秒でした。

step 1,000はtrain loss 4.444293、validation loss 5.593394、PPL 268.65、learning rate 2.9189e-4、経過109.36秒でした。step 1,500はtrain loss 4.536695、validation loss 5.346462、PPL 209.86、learning rate 2.6848e-4、経過166.33秒、step 2,000はtrain loss 3.786160、validation loss 5.110367、PPL 165.73、learning rate 2.3258e-4、経過220.45秒でした。step 2,500はtrain loss 3.421191、validation loss 4.980713、PPL 145.58、learning rate 1.8854e-4、経過278.24秒、step 3,000はtrain loss 3.565266、validation loss 4.889075、PPL 132.83、learning rate 1.4165e-4、経過330.51秒でした。step 3,100はtrain loss 3.711011、validation loss 4.839065、PPL 126.35、learning rate 1.3243e-4、経過343.84秒でした。step 3,100時点でNaN、OOM、shape errorはなく、validation lossは継続的に改善しています。学習は継続中です。

step 3,200はtrain loss 3.449614、validation loss 4.841980、PPL 126.72、learning rate 1.2337e-4、経過356.17秒でした。step 3,300はvalidation loss 4.817392、PPL 123.64、learning rate 1.1452e-4、経過367.39秒、step 3,400は4.793606、PPL 120.74、learning rate 1.0590e-4、経過379.51秒でした。step 3,500はtrain loss 3.090428、validation loss 4.780484、PPL 119.16、learning rate 9.7582e-5、経過389.53秒でした。

step 3,600はvalidation loss 4.788713、PPL 120.15、learning rate 8.9587e-5、経過402.06秒と一時的に悪化しましたが、step 3,700は4.753023、PPL 115.93、learning rate 8.1960e-5、経過410.87秒、step 3,800は4.737653、PPL 114.17、learning rate 7.4737e-5、経過421.85秒、step 3,900は4.717781、PPL 111.92、learning rate 6.7955e-5、経過433.29秒、step 4,000はtrain loss 3.027063、validation loss 4.696461、PPL 109.56、learning rate 6.1645e-5、経過444.79秒でした。step 4,000時点でもstep 3,500を除いてvalidation lossは改善傾向にあり、学習は継続中です。

step 4,100はtrain loss 3.394165、validation loss 4.680068、PPL 107.78、learning rate 5.5838e-5、経過455.34秒でした。step 4,200は4.674856、PPL 107.22、learning rate 5.0563e-5、経過467.05秒、step 4,300は4.658159、PPL 105.44、learning rate 4.5846e-5、経過478.74秒、step 4,400は4.651424、PPL 104.73、learning rate 4.1710e-5、経過489.00秒でした。step 4,500はtrain loss 3.034811、validation loss 4.649821、PPL 104.57、learning rate 3.8174e-5、経過499.85秒でした。

step 4,600はvalidation loss 4.651350、PPL 104.73、learning rate 3.5256e-5、経過512.27秒とわずかに悪化しましたが、step 4,700は4.634527、PPL 102.98、learning rate 3.2970e-5、経過522.85秒、step 4,800は4.616530、PPL 101.14、learning rate 3.1327e-5、経過534.40秒、step 4,900はtrain loss 2.914563、validation loss 4.607040、PPL 100.19、learning rate 3.0335e-5、経過545.92秒でした。step 5,000はtrain loss 3.380363、validation loss 4.615130、PPL 101.00、learning rate 3.0000e-5、経過555.90秒でした。step 4,900で最良validationを更新し、最終stepではわずかに戻りました。

学習はColab T4・PyTorch 2.11.0+cu128・CUDA 12.8・AMP有効で5,000 stepを完走しました。パラメータ数は50,207,616、GPU memoryは15,637,086,208 bytes、peak allocatedは1,528,071,680 bytes、peak reservedは1,591,738,368 bytes、summary上の総時間は556.84秒でした。Colab CLIの長時間実行ログ取得は途中でtimeoutしましたが、リモートのmetrics、summary、checkpointは正常に生成されており、学習結果には影響していません。最良checkpointはstep 4,900、validation loss 4.607039928436279、PPL 100.18714915065256です。best weightのSHA-256は`d61cf94bbdc053e7724f9648e935d3f0e4bc8a7ca5a6c2b0ec86acfd8137c508`、`best.json`は`2df0c9244a0b383c726275c290b1c473f4e20372901d38278c2d262a9c48eb9f`、`summary.json`は`71b58012ee3a5e755668e04ff2603f48a7763842680ac12da4a82880f4d0a2c0`、`metrics.jsonl`は`c27079a9a8669f3608e74e1fba365d914e12567fc5a54d52f5620d13272628bc`です。Colabから回収した軽量archiveは64ファイル、archive SHA-256は`d0c971c026690b014e2603ba344e9d5f764bb06d1fb852266f149e4e37096267`、checkpoint hash manifestは`/tmp/exp080-manifest.json`で、best checkpoint archiveのSHA-256は`f1e3db105a3380e2ccbae691db525e9d3e4d486118da9a5072d965c9dc0193cf`です。

固定promptの生成はstep 0では日本語と多数の崩れたTokenが混ざりましたが、step 1,000以降は`今日はなにをしていましたか?`の直後にEOSとなり、実質的に空応答へ崩れました。validation lossの改善だけでは自然な生成能力を意味しない典型例です。step 0からstep 5,000まで100 step間隔の生成本文は保存済みで、空応答も含めて省略していません。

次に、実験075と同じ5領域のvalidation、固定chat-test-v1 48例、生成全文を比較します。評価にはstep 4,900のbest checkpointを使い、080の評価設定・出力hash・075との差をこのノートへ追記します。

## 実験終了後の結果と解釈

実際に実行した評価コマンドは次のとおりです。

```bash
uv run python scripts/evaluate_torch.py domains \
  --config configs/issue1-both-50m-pretrain-10m-5k.toml \
  --checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-10m-5k/best.pt \
  --device cpu \
  --domain general=artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin \
  --domain conversation=artifacts/tokens/mixed-ja-80-10-10-v2-conversation-val.bin \
  --domain medical=artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin \
  --domain RPC=artifacts/tokens/issue1-real-persona-chat-validation.bin \
  --domain MRMP=artifacts/tokens/issue1-mrmp-validation.bin \
  --eval-batches 20 \
  --output artifacts/evaluations/issue1-both-50m-pretrain-10m-5k-domains.json
uv run python scripts/evaluate_torch.py chat \
  --config configs/issue1-both-50m-pretrain-10m-5k.toml \
  --checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-10m-5k/best.pt \
  --selection experiments/evaluation/chat-test-v1.json \
  --input artifacts/corpus/conversation-v1/test.jsonl \
  --device cpu --max-new-tokens 64 --seed 42 \
  --output artifacts/evaluations/issue1-both-50m-pretrain-10m-5k-chat-test-v1.json \
  --text-output artifacts/evaluations/issue1-both-50m-pretrain-10m-5k-chat-test-v1.txt
uv run python scripts/create_chat_review_template.py \
  --evaluation artifacts/evaluations/issue1-both-50m-pretrain-10m-5k-chat-test-v1.json \
  --output artifacts/evaluations/issue1-both-50m-pretrain-10m-5k-chat-review.json
```

075との差を同じ評価コードとseedで比較しました。5領域のvalidation lossは、generalが4.689170から4.607053へ-0.082117、conversationが2.889645から2.703728へ-0.185917、medicalが2.978035から2.732444へ-0.245591、RPCが2.855596から2.649845へ-0.205751、MRMPが2.582441から2.363312へ-0.219129となり、全領域で改善しました。特にmedicalと会話系の改善幅が大きく、10MコーパスへWikipediaを追加した構成と、Token予算をほぼ2倍にしたことは、言語モデルのvalidation性能には有効でした。

固定chat-test 48例では、080もEOS到達は48例中48例でした。平均生成Token数は9.8958から8.8750へ-1.0208と短くなりましたが、全体のToken overlap precisionは0.081253から0.110874へ+0.029621、recallは0.055935から0.067281へ+0.011346、F1は0.060641から0.074119へ+0.013478改善しました。short F1は0.046429から0.081483へ+0.035054、medium F1は0.067933から0.067581へ-0.000353、long F1は0.067563から0.073293へ+0.005731でした。079のように長く出すだけではなく、080は短いながら重複率が改善しており、少なくとも固定評価上は主目的に近い方向です。

一方、生成本文の質はまだ自然な会話と呼べる水準ではありません。事前学習の固定promptではstep 1,000以降にEOS直後の空応答へ崩れ、chat-testでも`コアラ!`、`こちらは、?`、`そうですね。どのくらいのお弁当ですー。`のように、局所的には日本語でも履歴への対応が弱い出力が残りました。したがって「10Mに増やせば自然な日本語になる」とは言えず、今回確認できたのは、事前学習Tokenを増やすことでvalidation lossと簡易chat F1が改善する可能性です。自然さをさらに高めるには、重複の少ない会話データを含む事前学習、文書境界とEOSの扱い、SFT前の学習量、そして生成時の停止挙動を分けて調べる必要があります。

評価JSON、生成全文、人手レビュー用JSONのSHA-256は、領域評価が`32a9d2008c684377881381dbfb1a03e65141bcbb89dad74c6b139b2de80fb684`、chat JSONが`fa6d6c2c99d77d90073175324a538897ee8ba7e3ba837fd1b5318dcc35df87e7`、生成TXTが`2f56602d25bf8977fd1131c0a254dad084f5c0ad1deea9b29ee4169d5087eded`、review JSONが`1b24becb7efbe36ee2ca797d82979957407c628b14be903bc2860ab3a5a4fcd3`です。Colabのcheckpoint hash manifestは`artifacts/checkpoints/issue1-both-50m-pretrain-10m-5k/colab_checkpoint_manifest.json`へ保存します。

今回の判断は、10M Token構成を主線の新しい基盤候補として採用する、です。ただし、source比率の変更とToken量の増加が同時に起きたため、改善の原因を一つに決めません。次は同じ075の5M Token列を2周以上見せる反復学習条件を実施し、080の「新しいデータを増やした効果」と比較します。これにより、データの多様性、反復回数、総Token予算のどれが効いたかを切り分けます。

## 次に試すこと

080で改善が確認できた場合は、同じ10M列でSFTへ進める前に、さらに学習Token数を増やすか、context length 512へ伸ばすかを比較します。改善が限定的な場合は、同じ5M列を複数周回する条件を実施し、データの多様性と反復学習の効果を分離します。どちらの場合も蒸留は主線へ入れず、教師なしの日本語事前学習で自然さがどこまで伸びるかを優先します。
