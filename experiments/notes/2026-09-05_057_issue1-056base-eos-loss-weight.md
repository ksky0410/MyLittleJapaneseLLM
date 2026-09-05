# 実験057：056の日本語基盤へ会話SFTを行い、EOS loss weightを比較する

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。Issue [#1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)で示された、一般日本語を保ちながら会話データを追加し、pretrainingと会話SFTを分けて評価する方針を継続します。実験056では、一般日本語53.282%、Wikipedia26.632%、会話6.651%、医療6.657%、青空文庫6.777%を含むToken列で、RoPE・LayerNorm・SwiGLUの20Mモデルを10,000 step学習しました。今回はそのbest checkpointを初期値にし、Issue #1の会話SFTと一般日本語rehearsalを行います。

今回の仮説は、response-only会話lossとrehearsal lossを半分ずつ混ぜる条件で、EOS tokenのloss重みを1.00から0.50へ下げると、短すぎる応答を避けて平均生成長と会話Token overlap F1が伸びる可能性がある、というものです。反対にEOSの停止信号が弱まり、過剰生成、反復、EOS到達率低下、general・medical loss悪化が起きる可能性もあります。EOSを除外した実験055では過剰生成とF1悪化が観測されたため、今回は0.00ではなく0.50を中間条件とし、同じbase checkpointからEOS weight 1.00と0.50を並列ではなく同一Colab session内で同じ順序に実行します。

## 条件と入力

設定は`configs/issue1-056base-rehearsal-ratio050-eos-ablation-colab-3k.toml`です。両条件でbase checkpoint、会話train/validation、rehearsal Token列、Tokenizer、モデル構造、seed、optimizer、学習率、stepを固定し、`--eos-loss-weight`だけを変更します。

- base checkpoint：`artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k/best.pt`、実験056 step 8,800、SHA-256 `476d848edd7566ff259ee74469912c5ad828a471a44bca1e53b20cd8bc571b21`
- 会話train NPZ：`artifacts/sft/chat-v1-context256/train.npz`、SHA-256 `400b8ffbc5b3752eaa16e003dab168c75e0a77046ac61c39630ef2409a73e609`
- 会話validation NPZ：`artifacts/sft/chat-v1-context256/validation.npz`、SHA-256 `5f52b3f4269e914184834d6e13d800604827abfd96f2b4c1ff5f665cd3f8f7b4`
- rehearsal Token列：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`
- Tokenizer：`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab 4,096、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`

モデルはdim 384、10層、6 heads、context length 256、RoPE、LayerNorm、SwiGLU、実測19,308,032 parametersです。学習はbatch size 8、3,000 step、evaluation/sample interval 100、checkpoint interval 500、AdamW、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 42です。rehearsal ratioは0.50、会話sampleは`<|speaker:DA|>`と`<|speaker:DC|>`を使います。

## 成功基準と再現情報

両条件がNaN、OOM、shape error、base checkpoint signature不一致なく完走し、metrics、stepごとの生成TXT、summary、periodic/best checkpoint metadataを回収できれば実装上の成功とします。品質比較では、会話validation loss、general・conversation・medical・RPC・MRMP loss、固定chat-test-v1のEOS到達率、平均生成Token数、Token overlap F1、出力の反復を比較します。医師国家試験由来のデータを含む評価結果を医学的正解や助言として扱いません。生成された文章は良否を問わず全件GitHubへ保存します。

学習前の基準commitは、EOS weight機能を固定した`8c92220`の後に、この実験のconfig・wrapper・package・concat・noteを追加した`c0393c9`です。bundleは77MBのbase checkpointとSFT/Tokenデータを含むため、45MB以下のpartへ分割してuploadし、Colab側でbytesとSHA-256を検証してから展開します。重い`.pt`本体はGitへ追加せず、Colab manifestとmetadataのhashを保存します。

実行前bundleは271,784,268 bytes、SHA-256 `66b9105cdce3a8169bb586ef176971c188c146b9595131b6d73b0bfa2d38e59a`です。6個の45MB以下のpartへ分割し、連結scriptにはこのbytesとhashを固定します。configのSHA-256は`73043053375a67210663684a999bc67340017936fd8486807374a933be7e4c2f`、`train_sft_torch.py`は`100c654d28fdd2817d8ea377588333802817f05c28a4a049aaf03942723fbbfc`、wrapperは`b0db52ba5b5bf044411830b012e50e40da8ec466e279ea3141e118fa925b8e23`、packageは`4788c7144087ecff5e503a163dc6970f86700fad85b456ee98da861fc9d59ce8`、concatは`9505b7bdf24d7877f877e838821fdf94039245d7184e1c63ca9ae6d73ca31dd1`です。base checkpointのhash、会話NPZ、rehearsal Token列、Tokenizerのhashはwrapperに固定済みです。

## 実験中の記録

開始前、bundle作成、part upload、連結hash、各条件の開始・途中・完了、成果物回収、評価、session停止を時系列で追記します。片方だけ完走した場合や、EOS weightの差が見えない場合も削除せず記録します。

2026-09-05、bundleは271,784,268 bytes、SHA-256 `66b9105cdce3a8169bb586ef176971c188c146b9595131b6d73b0bfa2d38e59a`で固定し、6個のpartをColabへuploadしました。連結scriptはbytesとSHA-256の両方を検証し、12個の入力hash照合後にEOS weight 1.00を開始しました。weight 1.00、続くweight 0.50はいずれも終了コード0で3,000 stepを完走しました。packageは80ファイル、archive 11,676 bytes、SHA-256 `91bc92d08890e51abd6fd6e6d49157e3cc2e64d1be6b8b811f079b78e3cf6725`でした。best checkpointのhashはEOS 1.00が`3f5cbe4bcf64e9186f6249dbd24ccb48b5f39fcd1e278144f81bd6e37d2fc903`、EOS 0.50が`9a28d6145a0d38ca4e6d5227ac3cdcf74836b8f6c488fe527e66586fecebe394`です。軽量成果物と両best重みはローカルへ回収済みです。

次に同じColab T4で、両条件をgeneral・conversation・medical・RPC・MRMPのdomain lossと固定chat-test-v1へ評価します。評価用Token列と会話入力のuploadは、056で確認済みの親ディレクトリ作成手順を使います。

評価用の`colab_evaluate_057.py`と`colab_prepare_eval_057.py`を追加し、同じT4 sessionへ評価コードとvalidation入力を送ります。評価時もbest checkpointのhashをmetadataで照合し、生成JSON/TXTを両条件で回収します。

評価の初回実行は、EOS 1.00条件のdomain評価開始後に終了コード1で停止しました。学習完了、checkpoint、入力uploadには影響していません。Colab wrapperが子プロセスstderrを表示しなかったため原因はこの時点では未確定で、stderrをそのまま表示する修正版へ更新して再実行します。

修正版評価scriptでstderrを回収した結果、057の学習bundleには学習自体に不要なgeneral validation Token列を含めていなかったため、評価時の`FileNotFoundError`が発生していました。SFT学習とcheckpointは正常で、general Token列を追加uploadして評価を再実行します。bundleのhashや学習条件は変更しません。

## 結果と解釈

実験終了直後に、両条件の実際のloss、PPL、学習時間、EOS到達率、生成例、055および056との差、仮説に対する判断を追記します。

## 次に試すこと

EOS weight 0.50が有望なら0.25または0.75を追加し、会話長と停止安定性の曲線を調べます。weight 1.00が優れるならEOSを弱めず、会話データのsource比率、speaker marker、rehearsal ratioを一つずつ比較します。SFTが安定した後に、056の20M構造を50Mへ拡大し、同じ日本語混合比率とIssue #1の会話データを保ったまま蒸留・reasoning SFTへ進みます。
