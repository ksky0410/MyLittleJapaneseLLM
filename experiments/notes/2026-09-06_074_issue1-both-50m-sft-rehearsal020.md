# 実験074：50M both-SFTへ一般日本語rehearsal 0.20を適用

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。実験073では、Issue #1で候補にしたRPC（RealPersonaChat）とMRMP（Multi-Relational Multi-Party Chat Corpus）を含む`both`事前学習データを、約1M Token・約5.12M exposure Tokenで50,207,616 parameterのモデルへ学習しました。しかし20M基盤と同じ条件で比較すると、50Mのgeneral・conversation・medical・RPC・MRMP validation lossはすべて悪化しました。この結果は、約1M Tokenに対して容量だけを増やすとデータ不足または過学習が早く現れるという仮説を支持します。

今回は実験073のbest checkpointへ、実験067で標準候補にしたresponse-only SFTと一般日本語rehearsal ratio 0.20を適用します。仮説は、会話SFTによる応答形式の獲得は50Mでも起こり、20Mの同一SFT条件より会話生成の自然さ・短中長別F1・EOS到達率が改善することです。一方、事前学習で確認されたデータ不足の影響が強ければ、50Mは20MよりSFT validation lossが下がらず、容量を増やす前にpretraining Token数を増やす必要がある可能性があります。

比較は、20Mの実験067相当の`issue1-both-20m-sft-source-rehearsal020-mps-3k`と、今回の50M `issue1-both-50m-sft-source-rehearsal020-colab-3k`で行います。SFT train・validation、rehearsal Token列、Tokenizer、seed、learning rate、3,000 step、rehearsal ratio、EOS loss weightを揃え、モデル容量とbase checkpointだけを変数にします。自動評価だけでなく、48例の生成本文、人手レビュー用テンプレート、general・conversation・medical・RPC・MRMPの領域別lossを保存します。医師国家試験由来のデータは一般日本語モデルの医療領域保持を確認するために使いますが、医療専用モデルにはしません。元の`/Users/koseki/projects/medilink_analysis`とその原データは変更・削除しません。

## 再現条件

実験開始時点の基準commitは、実験073の評価までを含む`b671720`です。074のconfigは[`configs/issue1-both-50m-sft-source-rehearsal020-colab-3k.toml`](../../configs/issue1-both-50m-sft-source-rehearsal020-colab-3k.toml)、Colab実行wrapperは[`scripts/colab_bootstrap_074.py`](../../scripts/colab_bootstrap_074.py)、軽量成果物回収scriptは[`scripts/colab_package_074.py`](../../scripts/colab_package_074.py)です。学習コードは既存の[`scripts/train_sft_torch.py`](../../scripts/train_sft_torch.py)を使い、074専用の出力先へ保存します。

configのSHA-256は`c5c877deefdba7fa4c0256c908f7ad4990578650a351b7bc1835d66b0cff8552`、`train_sft_torch.py`のSHA-256は`c566b18a851d0a561e98e3123a03365a97c781c6617843393db388c24819b31b`、074 wrapperのSHA-256は`ea075ecc8c2434c40a38cf5db5090335a000fb85c4e45949fc284778a3aeeec3`、package scriptのSHA-256は`0d4e1acab12994b61e5d5850a07ba59c77790e88041502bb6b1197af0dd6b0bc`です。

base checkpointは実験073の[`best.pt`](../../artifacts/checkpoints/issue1-both-50m-pretrain-mps-2p5k/best.pt)で、step 1,300、実測parameter数50,207,616、SHA-256は`6f555eeb7e3dbc2bab925d7f868444c40edc7f0c1dee1a54c7c03c91d61a2503`です。会話SFT trainは64,423例・response 770,990 Token、`artifacts/sft/issue1-both-balanced-v1/train.npz`、SHA-256は`645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`です。validationは49,045例・response 738,660 Token、`artifacts/sft/issue1-both-full-v1/validation.npz`、SHA-256は`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`です。rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`、Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

モデルはdim 576、12層、9 heads、context length 256、vocab 4,096、RoPE、LayerNorm、SwiGLUです。学習はbatch size 8、3,000 step、eval・sample interval 100、checkpoint interval 500、eval batches 20、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 42、EOS loss weight 0.50、rehearsal ratio 0.20、cosine schedule終点3,000 stepで実行します。Colab GPUでは`--device auto`、MPS fallbackでは`--device mps`を指定します。生成はconversation形式で、話者AをDA、話者BをDC、最大160 Token、temperature 0.8、top-k 40とします。

Colabで使う実行コマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-50m-sft-source-rehearsal020-colab-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-mps-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-50m-sft-source-rehearsal020-colab-3k \
  --samples-dir artifacts/samples/issue1-both-50m-sft-source-rehearsal020-colab-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device auto
```

Colab bundleは074専用の分割bundleとして作成します。50M base checkpoint、metadata、SFT train・validation NPZ、rehearsal Token列、Tokenizer、config、`src` package、学習script、wrapper、package scriptを含めます。元の会話JSONL、医師国家試験の原本、`medilink_analysis`のディレクトリは含めません。bundleのサイズ・SHA-256、分割数、Colab側で連結後に検証した値を、学習開始前にこのノートへ追記します。

## 成功・失敗の判定基準

3,000 stepをNaN、OOM、shape errorなく完走し、best checkpoint metadata、metrics、summary、step 0〜3,000の生成本文を保存できれば学習実験として成功とします。Colabを使った場合はGPU種別、Torch・CUDA情報、peak memory、軽量archiveのmanifest、best checkpointのhashを回収します。完走後は共通5領域評価、固定chat-test 48例、人手レビュー用JSONを作成します。SFT validation lossとF1が20M基盤を上回ることは性能上の成功条件ですが、未達でも失敗を削除せず、データ量を増やす次の仮説へつなげます。

学習前に最低限の予定を記録し、学習中は少なくとも100 step間隔のmetricsと生成文を保存します。500 stepごとにcheckpoint metadataとノートを確認し、GitHubへcommit・pushします。SFT loss、rehearsal loss、総合loss、validation loss、PPL、learning rate、経過時間、メモリ、警告、途中停止を記録します。生成文は品質に関係なく全件Gitで追跡し、空出力・EOS直後・特殊Token混入・話題逸脱も削除しません。

## 実験中の記録

準備時点ではColab sessionは空で、`colab sessions`でactive sessionがないことを確認しました。次に074専用bundleを作成し、GitHubへpushした後にT4割当を試します。割当がHTTP 503や上限エラーで失敗した場合は、エラー内容とsession状態を記録して、同じ入力・seed・出力先のMPS実行へ切り替えます。Colab upload、bundle hash検証、学習開始、500 stepごとの節目、成果物回収、session停止を時系列で追記します。

## 実験終了後の結果と解釈

学習終了後に、実際のruntime、parameter数、最良・最終loss、学習時間、最大メモリ、生成本文の代表例、checkpoint・入力・bundleのhash、20Mとの差を追記します。領域別lossとchat-testの自動指標が一致しない場合は、片方だけで容量効果を断定しません。人手レビュー未実施の場合はその状態を明記します。

## 次に試すこと

50M SFTが完走した場合は、20Mとの容量比較を終えた後、同じ50M構造でpretraining Token数を増やす条件を試します。今回のSFTでも会話品質が伸びない場合は、長文samplingへ戻る前にデータ量、SFTとrehearsalのToken予算、学習率scheduleを一つずつ分離します。Colabが安定して利用できる場合は、次の長時間pretrainingや50M以上のモデルへ移行し、同じノートと成果物管理を維持します。
