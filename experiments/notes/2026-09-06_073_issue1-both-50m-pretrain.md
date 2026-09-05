# 実験073：Issue #1のboth基盤を約50Mへ拡張

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。GitHubの[Issue #1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)は、標準的な日本語だけでは不足しやすい現代の雑談表現、相づち、短文、砕けた語尾、話者交代、複数人会話、話題継続を検証する候補です。これまでRPC（RealPersonaChat）とMRMP（Multi-Relational Multi-Party Chat Corpus）を含む`both`条件を20Mモデルで検証してきました。医師国家試験由来のデータは一般モデルの医療領域保持を測るために利用しており、医療専用化は行いません。元の`/Users/koseki/projects/medilink_analysis`とその原データは絶対に変更・削除しません。

実験050で作成した`both`の約1M Token列を用いた20M checkpointは、general・conversation・medical・RPC・MRMPの評価用基盤になっています。実験067では20M checkpointへ会話SFTとrehearsalを適用し、実験068〜072では長文応答samplingを比較しました。しかし長文2/6条件はseed 42・123・777で生成F1の分散が大きく、採用を保留しました。今回はsampling条件を変えず、まず事前学習モデルの容量だけを20Mから約50Mへ拡張します。Issue #1の会話能力を50Mで評価するための初期値を作ることが目的です。

モデルは現行20Mと同じTokenizer、RoPE、LayerNorm、SwiGLU、context length 256を維持し、dim 576、12層、9 headsへ拡張します。head dimensionは64で、パラメータ数は約50,163,840です。20M基盤と同じ`issue1-both-1m-fineweb-train.bin`、同じgeneral validation、batch size 8、2,500 step、約5.12M exposure Token、AdamW、learning rate 3e-4から3e-5、warmup 300、weight decay 0.1、seed 42を使います。これにより、容量差とデータ差を分離します。

仮説は、50Mモデルでは20Mモデルよりgeneral validation lossが下がり、会話・RPC・MRMPのvalidation lossも改善することです。事前学習だけでは会話応答の形式やEOSを十分に獲得しない可能性があるため、固定chat-testの自然さが直ちに改善するとは仮定しません。完走後、同じ50M checkpointへ標準的なresponse-only SFTを適用する実験へ進み、20Mの067相当条件と容量差を比較します。学習が長時間またはMPSメモリ不足で失敗した場合も、そのまま記録します。

## 再現条件

実験072のseed sweepまで完了し、origin/mainへpush済みの基準commitは`c78a01e`です。本実験の設定ファイルは[`configs/issue1-both-50m-pretrain-mps-2p5k.toml`](../../configs/issue1-both-50m-pretrain-mps-2p5k.toml)です。モデル構造はRoPE・LayerNorm・SwiGLU、dim 576、12層、9 heads、context length 256、vocab 4,096、約50,163,840 parametersです。Tokenizerは`mixed-ja-80-10-10-v2-unigram.model`です。

設定ファイルのSHA-256は`b16ed5b4b40c621658d3d60a31334fa33331d648fd4c4483c3d67bd2567ceede`です。`scripts/inspect_model.py`の概算は50,163,840でしたが、PyTorchモデルを実際に生成して数えると50,207,616でした。差分はSwiGLUのLinear layerに含まれるbias分であり、学習時のcheckpoint metadataに記録された実測値50,207,616を正式値として扱います。

学習データは`artifacts/tokens/issue1-both-1m-fineweb-train.bin`、general validationは`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`です。元データそのものではなく、実験050で作成済みのToken列を読み込みます。Token列のsource mixtureと約999,970 Tokenという規模を20M基盤と揃えます。入力のSHA-256は、Tokenizerが`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`、train Token列が`758b46f6bb946afd7e2c3604714db71166d79564f8c652e8cc950b23d3338879`、general validationが`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。

学習条件はbatch size 8、2,500 step、eval・sample interval 100、checkpoint interval 500、eval batches 20、learning rate 3e-4から3e-5、warmup 300、weight decay 0.1、seed 42です。生成promptは`今日なにしてた？`、最大160 Token、temperature 0.8、top-k 40です。MPSではAMPを無効にします。

再現に使うコマンドは次のとおりです。

```bash
uv run python scripts/train_torch.py \
  --config configs/issue1-both-50m-pretrain-mps-2p5k.toml \
  --device mps
```

学習開始前に設定ファイルのSHA-256を記録します。step 0から100 step間隔の生成本文、metrics、500 step間隔のcheckpoint metadataを保存し、500 stepごとにcommit・pushします。重い`.pt`本体はGit管理外ですが、metadataへSHA-256を残します。Colab CLIでT4割り当てを先に試し、失敗時はHTTP応答とsession状態を記録してMPSへ切り替えます。

## 成功・失敗の判定基準

2,500 stepをNaN、OOM、shape errorなく完走し、step 0〜2,500の生成本文、metrics、summary、checkpoint metadataを保存できれば学習実験として成功とします。性能面では、20M基盤の同じ5領域と比較し、generalだけでなくconversation・medical・RPC・MRMPの変化を確認します。固定chat-testは事前学習checkpointの会話応答限界を測る参考値として保存しますが、SFT後の品質と混同しません。悪い生成や途中失敗は削除しません。

## 実験中の記録

この節にはColab試行、MPS切り替え、開始時の実測parameter数、100 stepごとのmetrics、500 stepごとの生成本文とcheckpoint metadata、警告、メモリ問題、途中停止を時系列で追記します。原文データや医師国家試験の原本はbundleやGitへ追加しません。

2026年9月6日、MPS学習の開始前に`colab new -s exp073-both-50m-pretrain --gpu T4`を実行しました。しかしColab CLIのassignment endpointがHTTP 503 `Service Unavailable`を返し、セッション作成に失敗しました。直後の`colab sessions`は`No active sessions found on server.`でした。bundle uploadやColab上の学習は発生していないため、同一条件をMPSで実行します。

同日、Colab失敗を記録したcommit `050572d`の後、MPSで学習を開始しました。開始時の実測parameter数は50,207,616でした。step 1はtrain loss 8.849134、validation loss 8.818872、PPL 6760.6331、learning rate 1.0000e-6、経過時間6.57秒でした。step 100はvalidation loss 7.162676、PPL 1290.3593、step 200は7.153232、PPL 1278.2307、step 300は6.894857、PPL 987.1843、step 400は6.770894、PPL 872.0912となりました。step 500ではtrain loss 4.698257、validation loss 6.559118、PPL 705.6490、learning rate 2.9459e-4、経過時間542.69秒となりました。step 0〜500のmetrics、checkpoint metadata、生成本文を保存しました。step 500の生成は日本語らしい助詞や語尾がまだ崩れており、`今日なにしてた?がちいてているだけです。`のような断片が多く見られますが、step 1よりlossは明確に下がっています。ここまでNaN、OOM、shape error、警告はありません。学習は継続中です。

## 実験終了後の結果と解釈

学習終了直後に、backend、Torchバージョン、device、AMP、parameter数、学習時間、最良checkpoint、最終および最良loss、生成本文、入力hash、成果物hashを追記します。20M基盤との差から容量の効果を評価し、次の50M SFT実験の初期checkpointを確定します。

## 次に試すこと

完走した場合は、この50M基盤へ実験067相当の標準response-only SFTとrehearsal ratio 0.20、EOS loss weight 0.50を適用します。会話生成の改善が見えなければ、長文samplingへ戻る前に人手レビューと話題適合評価を追加します。MPSの速度またはメモリが厳しい場合は、Colab T4が復旧した時点で同じconfigを再実行できるようにします。
