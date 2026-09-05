# 実験056：同一データでRoPE・SwiGLUの20Mモデルを比較する

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。GitHub Issue [#1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)を再確認しました。Issue #1は現在もOpenで、RealPersonaChatとMRMPを既存の一般日本語へ混ぜること、通常の事前学習と会話形式SFTを分けること、固定prompt・出所・ライセンス・話者境界を記録することを求めています。実験048〜055で、会話sourceの追加、pretraining、rehearsal SFT、会話prompt、EOS損失を順に検証できたため、Issue #1は今後も継続する実験候補として正式に扱います。

今回は医療専用モデルを作る実験ではありません。Issue #1の方針に沿い、一般日本語を主軸に、会話データと医師国家試験由来の加工済みデータを含む既存の混合Token列をそのまま使います。混合Token列のsource比率は青空文庫6.777%、FineWeb2 Edu Japanese 53.282%、Wikipedia 26.632%、会話6.651%、医療6.657%です。元の`/Users/koseki/projects/medilink_analysis`と、その中の医師国家試験データは読み取り専用で扱い、変更・移動・削除を行いません。

実験041では、同じ7.5M Token列と20M級の学習予算で、absolute position embedding・LayerNorm・GELUを使いました。今回の仮説は、データ、Tokenizer、モデル規模、context、batch、optimizer、学習率、seed、総stepを固定し、位置表現だけをabsoluteからRoPEへ、FFNだけをGELUからSwiGLUへ置き換えると、現代的なdecoder-only構造の方がgeneral validation lossと日本語生成の安定性で有利になる可能性がある、というものです。一方で、20M級・約20M学習Tokenという小規模条件では差が誤差に埋もれる可能性も高く、悪化やEOS停止率の低下も含めて記録します。

## 使用する入力とモデル

設定は`configs/fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k.toml`です。学習はColab T4上のPyTorch/CUDAで行います。入力は次のとおりです。

- 学習Token列：`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin`、7,499,997 Token、SHA-256 `3bad9f5f9546d98fc598d602a053648679d6e7817161f0add7a219b020c7440a`
- general validation Token列：`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、SHA-256 `c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`
- Tokenizer：SentencePiece Unigram、vocab 4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- 混合本文manifest：`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1.manifest.json`、出力本文SHA-256 `6a0ea02b8d0e5baf83c05e1d487c53fce6e9b8b1056d06c440caa546948aca31`

モデルはdim 384、10層、6 heads、context length 256、MLP倍率4、LayerNorm、RoPE、SwiGLUです。SwiGLUの中間次元は実装上1,024で、実測parameter数は19,308,032となる見込みです。比較対象の実験041はabsolute position embedding・GELUで19,401,216 parametersでした。差は約0.5%なので、今回の比較ではモデル規模の差を併記し、構造変更の効果を厳密に証明したとは扱いません。

## 学習条件と成功基準

batch size 8、最大10,000 step、evaluation interval 100、sample interval 100、checkpoint interval 1,000、evaluation batches 20、AdamW、learning rate 3e-4から3e-5、warmup 300、weight decay 0.1、seed 42を使います。1 stepあたり2,048 Token、学習総Token exposureは約20.48Mです。固定promptは`今日は`、最大生成長160、temperature 0.8、top-k 40です。

成功基準は、NaN、OOM、shape error、Token列不足なく10,000 stepを完走し、metricsを100 step間隔、生成文をstep 0から100 step間隔、periodic checkpointを1,000 step間隔で保存・回収できることです。品質比較では、実験041のgeneral loss 4.5294143359、実験041のconversation・medical loss、固定chat-test-v1のEOS到達率・平均生成長・Token overlap F1を参照します。lossだけで会話能力や医学的正確性を主張しません。生成文は良いものも崩れたものも省略せず、GitHubで追跡します。

## 再現コマンド

開始前にこのノート、config、Colab wrapper、成果物package scriptをcommit・pushし、そのcommitのファイルからbundleを作成します。入力bundleは次のコマンドで作成し、bytesとSHA-256を実行前にノートへ追記します。

学習前の基準commitは`2819611`です。configのSHA-256は`b382e890e0cda18db24754662d6a30b8e4fb802092e58b20cc3c3654dd65007d`、`scripts/train_torch.py`は`c8fb40406ec74635ba63159f86fcd55ef71724edc7cb8ffda53453222640203e`、wrapperは`d57e2141c097229d8528a7e28f1661552e104a1c6380e4c2edf50f8acac48691`、package scriptは`8b9917f0e5a98787f3013c127cfdd0b4bcfe248f72847e651b4c78c61c4162a6`です。送信用bundleは`/tmp/exp056_bundle.tar.gz`、9,989,309 bytes、SHA-256 `809026d7511a6ebb141e4ad7483f1c24ad7a806164082da7baae3e947dcc5066`です。bundleにはPython cacheを含めていません。wrapper内では、これらに加えてモデル実装・入力Token列・Tokenizerのhashを照合します。

```bash
tar -czf /tmp/exp056_bundle.tar.gz \
  src/my_little_japanese_llm \
  scripts/_common.py \
  scripts/train_torch.py \
  configs/fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k.toml \
  artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin \
  artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin
colab new --session exp056-20m-modern-architecture --gpu T4
colab upload --session exp056-20m-modern-architecture /tmp/exp056_bundle.tar.gz /content/exp056_bundle.tar.gz
colab exec --session exp056-20m-modern-architecture --timeout 3600 --file scripts/colab_bootstrap_056.py
colab exec --session exp056-20m-modern-architecture --timeout 120 --file scripts/colab_package_056.py
colab download --session exp056-20m-modern-architecture /content/exp056-lightweight.tar.gz /tmp/exp056-lightweight.tar.gz
colab download --session exp056-20m-modern-architecture /content/exp056-manifest.json /tmp/exp056-manifest.json
colab stop --session exp056-20m-modern-architecture
```

wrapperはbundle展開先を`/content/small_llm_056`へ固定し、必要な実行コード・入力ファイルのSHA-256を照合してから学習を始めます。学習後は軽量archive、manifest、best checkpointを回収します。重い`.pt`本体はGitへ追加しませんが、Colab manifestとcheckpoint metadataのSHA-256をノートへ残します。

## 実験中の記録

開始前の計画、コードcommit、bundle hash、Colab割当、学習中の節目、回収結果をこの節へ時系列で追記します。途中停止、割当失敗、入力hash不一致、生成の崩れも削除せず記録します。

2026-09-05、開始前にcommit `2819611`を基準としてbundleを作成し、bundleは9,989,309 bytes、SHA-256 `809026d7511a6ebb141e4ad7483f1c24ad7a806164082da7baae3e947dcc5066`で固定しました。Colab session `exp056-20m-modern-architecture`の新規T4割当に成功し、probeはPython 3.13.15、PyTorch 2.11.0+cu128、CUDA 12.8、Tesla T4、15,360 MiBを報告しました。matmul probeは0.1659891680秒、最大確保メモリは75,628,544 bytesでした。bootstrapはconfig、学習コード、モデル実装、Tokenizer、学習・検証Token列の11 hashを照合し、すべて一致してから学習を開始しました。

学習は終了コード0で完走しました。Colab packageの軽量archiveには114ファイルが含まれ、archiveは31,938 bytes、SHA-256 `e02ac1cf35f7aba2f2448627d2536fbff97be3de4e219547f7dc82b6d2105e29`です。manifestのSHA-256は`a44be25e71902e0e9284d8d1f5c423216c04894f5081de97b3369841099e8fdb`、best checkpointは77,267,142 bytes、SHA-256 `476d848edd7566ff259ee74469912c5ad828a471a44bca1e53b20cd8bc571b21`です。学習終了後もbest checkpoint本体を`/tmp/exp056-best.pt`へ回収し、重い重みはGitへ追加せずhashだけを保持します。生成TXT、metrics、summary、checkpoint metadata、Colab manifestはリポジトリへ展開して追跡します。

学習完了後、同じColab T4上でdomain評価とheld-out chat-test-v1評価を開始します。評価用に`evaluate_torch.py`、`evaluate_chat_dataset.py`、5種類のvalidation Token列、固定48例の会話入力とselection manifestを使用し、general・conversation・medical・RPC・MRMPのloss、EOS到達率、平均生成長、Token overlap F1を保存します。

会話JSONLとselection manifestの初回uploadは、Colab側の親ディレクトリが未作成だったためHTTP 500で失敗しました。Token列4種類のuploadや学習成果物には影響しておらず、専用の`colab_prepare_eval_056.py`で保存先を作成してから再送します。

best checkpointの評価に加え、実験041との学習stepをそろえた比較が必要なため、step 10,000の`step_010000.pt`も同じdomain・chat評価へ通します。bestがstep 8,800であること自体を隠さず、両方の結果を別ファイルに保存します。

## 結果と解釈

実験終了直後に、実際のruntime、最終・最良loss、PPL、checkpoint、生成例、実験041との差、仮説と一致した点、次に試す変更を追記します。未実施の場合は、未実施の理由と次の確認方法を明記します。

2026-09-05、学習は10,000 stepまで終了しました。実測parameter数は19,308,032、最良checkpointは`best.pt`のstep 8,800、general validation lossは4.3458760579、perplexityは77.159604です。最終step 10,000のgeneral validation lossは4.3502650261、perplexityは77.498999です。学習時間は637.072926秒、Colab runtimeはPython 3.13.15、PyTorch 2.11.0+cu128、CUDA 12.8、Tesla T4、AMP有効でした。peak memory allocatedは787,465,728 bytes、reservedは834,666,496 bytesです。実験041のabsolute+GELU条件は19,401,216 parameters、学習時間428.795688秒、peak allocated 689,949,184 bytesでしたので、056は約1.49倍の時間と約1.14倍のpeak memoryを使いました。RoPEの位置計算を各attention層で行う現在の実装が速度差の一因である可能性があり、性能だけでなく実装効率も今後の改善対象です。

best checkpointのdomain評価では、general 4.345876、conversation 2.610106、medical 2.482179、RPC 2.504567、MRMP 2.254911でした。実験041の同じToken列に対するgeneral 4.529341、conversation 2.703211、medical 2.641884と比べると、それぞれ0.183465、0.093106、0.159705低下しました。RPCとMRMPも評価可能にした今回の結果では、Issue #1で追加した会話sourceへ対応するlossがRPC 2.504567、MRMP 2.254911となりました。ただし、このloss低下は会話内容の正確な応答や医学的正確性を意味しません。best domain評価JSONは`artifacts/evaluations/fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k-domains.json`、SHA-256は`307f0b6cc05887e77745fe609440132ecfe2ca660b259b4170f9976b57100f0e`です。

固定chat-test-v1のbest checkpointは48/48例でEOSへ到達し、平均生成Token数9.6042、Token overlap F1は0.101295でした。実験041の48/48、平均8.5417、F1 0.086434よりF1は0.014861高くなりました。short・medium・longのF1はそれぞれ0.128008、0.104347、0.071531です。一方、step 10,000へそろえたcheckpointではEOS 48/48、平均9.3542、F1 0.085153となり、041のF1をわずかに下回りました。したがって、RoPEとSwiGLUは今回の条件でvalidation lossを明確に改善し、best checkpointの固定chat F1も改善しましたが、最終stepのchat F1まで一貫して改善したとは言えません。評価JSONはbestが`artifacts/evaluations/fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k-chat-test-v1.json`、finalが`artifacts/evaluations/fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k-final-chat-test-v1.json`です。

固定prompt `今日は`の生成は、step 0では「媒可能性格ブ…」のような記号・文字の崩れが目立ち、step 5,000でも特殊記号が多く残りました。bestのstep 8,800では「お客さま」「お問い合わせ」など日本語のまとまりは現れましたが、Web広告・URL風の断片が混ざりました。step 10,000では`Phone`、`Photo`、`WordPress`、`Ruby`など英数字を含む技術文書風の断片となり、自然な短い物語や安定した会話にはなっていません。これらを含むstep 0からstep 10,000までの生成TXTは`artifacts/samples/fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k/`にすべて保存しています。良く見える出力だけを選別していない点を明記します。

事前の仮説のうち、同じデータと学習予算でRoPE・SwiGLUへ変更するとgeneral、conversation、medicalのlossが下がるという部分は支持されました。反対に、lossの改善が自然な生成へそのままつながるとは確認できず、速度とmemoryには明確なコストがありました。構造変更は有望な基盤候補として採用しますが、今回一回のseedだけでRoPE・SwiGLU全般の優位性を結論づけません。Colab評価後にsession `exp056-20m-modern-architecture`を停止し、`colab sessions`が空であることも確認しました。初回の評価入力upload失敗はノートへ残し、ディレクトリ作成後に再送して評価を完了しました。

## 次に試すこと

この実験でRoPE・SwiGLUが有望なら、同じ構造のまま学習Token予算を増やすか、RMSNormまたはGQAを一要素ずつ追加します。構造差が小さい場合は、Issue #1の会話source比率とSFTのデータ形式へ戻り、一般日本語・会話・医療を混ぜるpretrainingと会話SFTを別々に比較します。安定した基盤ができた時点で、50M級への拡大と日本語reasoning蒸留へ進みます。

今回の結果を受けた次の第一候補は、056のbest checkpointを初期値として、Issue #1の会話データへresponse-only SFTとrehearsal lossを適用する実験です。EOS loss weightも含めて、会話F1だけでなくgeneral・medical・RPC・MRMP lossと生成の過剰継続を比較します。その後、RoPEの位置計算をcacheして速度を改善し、同一seedで再実行するか、RMSNormまたはGQAを一つだけ追加します。データ量を増やす場合も、まず一般日本語・会話・医療のsource比率と重複を固定してから、50M級へ拡大します。
