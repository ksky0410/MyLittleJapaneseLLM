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

step 600ではvalidation loss 6.493899、PPL 661.0962、step 700では6.425177、PPL 617.1899、step 800では6.318794、PPL 554.9033、step 900では6.275675、PPL 531.4853となりました。step 1,000ではtrain loss 3.407382、validation loss 6.241747、PPL 513.7555、learning rate 2.3815e-4、経過時間1167.69秒となりました。step 600〜1,000の生成本文、step 1,000のcheckpoint metadata、metricsを保存しました。step 1,000の固定prompt生成はprompt直後にEOSへ到達して本文が空であり、事前学習途中の会話応答能力はまだ成立していません。validation lossは継続して改善し、ここまでNaN、OOM、shape error、警告はありません。学習は継続中です。

step 1,100ではvalidation loss 6.267543、PPL 527.1804、step 1,200では6.264282、PPL 525.4644、step 1,300では6.228799、PPL 507.1459、step 1,400では6.230421、PPL 507.9691となりました。step 1,500ではtrain loss 3.220841、validation loss 6.232976、PPL 509.2687、learning rate 1.4598e-4、経過時間1796.26秒となりました。step 1,100〜1,500の生成本文、step 1,500のcheckpoint metadata、metricsを保存しました。step 1,500の固定prompt生成は`今日なにしてた? ここにも知りませんでした ⁇`で、学習途中として日本語の断片は増えましたが、質問への適切な応答にはなっていません。step 1,300でvalidationが一時改善した後、1,400〜1,500で横ばいとなっています。ここまでNaN、OOM、shape error、警告はありません。学習は継続中です。

step 1,600ではtrain loss 3.219235、validation loss 6.236125、PPL 510.8750、step 1,700ではvalidation loss 6.247959、PPL 516.9569、step 1,800では6.274849、PPL 531.0464となりました。step 1,900ではtrain loss 2.006298、validation loss 6.274279、PPL 530.7437でした。step 2,000ではtrain loss 2.943165、validation loss 6.258281、PPL 522.3204、learning rate 6.3100e-5、経過時間2785.67秒となりました。step 1,600〜2,000のmetrics、生成本文（[`step_001600.txt`](../../artifacts/samples/issue1-both-50m-pretrain-mps-2p5k/step_001600.txt)、[`step_001700.txt`](../../artifacts/samples/issue1-both-50m-pretrain-mps-2p5k/step_001700.txt)、[`step_001800.txt`](../../artifacts/samples/issue1-both-50m-pretrain-mps-2p5k/step_001800.txt)、[`step_001900.txt`](../../artifacts/samples/issue1-both-50m-pretrain-mps-2p5k/step_001900.txt)、[`step_002000.txt`](../../artifacts/samples/issue1-both-50m-pretrain-mps-2p5k/step_002000.txt)）、step 2,000のmetadata（[`step_002000.json`](../../artifacts/checkpoints/issue1-both-50m-pretrain-mps-2p5k/step_002000.json)）を保存しました。step 1,600では外来語や英字を含む記事風の断片、step 1,700〜1,900では「社会」「企業」「コーヒー」などを含む文の断片、step 2,000では「記事」「Q&A」「アルゴリズム」などを含む列挙風の断片が出ました。日本語らしい文字列の連続性はstep 500より改善していますが、promptの質問に答える会話応答にはまだなっておらず、事前学習だけではIssue #1の会話品質を判断できません。validation lossはstep 1,300の6.228799を底に悪化し、step 2,000でも6.258281に留まっているため、現時点では50M化だけで20M基盤を上回ると判断せず、完走後の同一SFT条件で比較する必要があります。ここまでNaN、OOM、shape error、警告はありません。学習は継続中です。

step 2,100ではtrain loss 2.400643、validation loss 6.305353、PPL 547.4949、step 2,200では2.483952、6.307060、548.4301、step 2,300では2.094286、6.322453、556.9372、step 2,400では2.455391、6.347757、571.2100となりました。step 2,500ではtrain loss 2.224194、validation loss 6.365837、PPL 581.6313、learning rate 3.0000e-5、経過時間3978.46秒で最終評価が完了しました。step 2,100〜2,500の生成本文（[`step_002100.txt`](../../artifacts/samples/issue1-both-50m-pretrain-mps-2p5k/step_002100.txt)、[`step_002200.txt`](../../artifacts/samples/issue1-both-50m-pretrain-mps-2p5k/step_002200.txt)、[`step_002300.txt`](../../artifacts/samples/issue1-both-50m-pretrain-mps-2p5k/step_002300.txt)、[`step_002400.txt`](../../artifacts/samples/issue1-both-50m-pretrain-mps-2p5k/step_002400.txt)、[`step_002500.txt`](../../artifacts/samples/issue1-both-50m-pretrain-mps-2p5k/step_002500.txt)）を保存しました。step 2,100では人物・職場・世界などの断片、step 2,200ではホームページ・広告・アプリケーションなどの記事断片、step 2,300では短い未完文、step 2,400では日本語と英字が混ざるサイト説明風の文、step 2,500では年金制度・価値観・イベント・アルゴリズムを混ぜた列挙風の文が出ました。最終的にも日本語の連続した文字列は生成できますが、固定promptへの自然な返答にはなっていません。

## 実験終了後の結果と解釈

2026年9月6日、step 2,500までNaN、OOM、shape errorなく完走しました。backendはPyTorch、Torchは2.14.0、deviceはMPS、AMPは無効、実測parameter数は50,207,616でした。総経過時間はsummary上3982.70秒（約66.4分）です。最大メモリ使用量と温度は今回の実行ログでは取得できなかったため、未計測と記録します。最良checkpointはstep 1,300の`best.pt`で、train loss 3.637268、validation loss 6.228799、PPL 507.1459でした。`best.pt`のサイズは200,870,286 bytes、SHA-256は`6f555eeb7e3dbc2bab925d7f868444c40edc7f0c1dee1a54c7c03c91d61a2503`です。最終step 2,500はtrain loss 2.224194、validation loss 6.365837、PPL 581.6313であり、最良stepからvalidation lossが0.137038悪化しました。学習率を下げてもstep 1,300以降のvalidationは改善せず、約1M Tokenの同一学習データで50Mへ容量だけを増やすと、今回の条件では過学習またはデータ不足が早く現れる結果となりました。

固定prompt `今日なにしてた？`では、step 500の断片的な日本語からstep 2,500の記事・制度・広告に似た列挙風の出力まで変化しましたが、質問への応答形式や話題の整合性は獲得できませんでした。したがって、事前学習だけのchat生成品質については成功基準を満たしていません。ただし、50Mモデルの学習・checkpoint・100 step間隔のmetricsと生成文保存は成功しており、容量比較用の初期checkpointとしては利用可能です。次に同じbest checkpointのgeneral・conversation・medical・RPC・MRMP領域評価とchat-testを実行し、20M基盤との差を数値で確認します。評価結果を追記するまでは、50Mの総合性能向上を主張しません。

## 次に試すこと

領域別評価とchat-testを終えた後、この50M基盤へ実験067相当の標準response-only SFTとrehearsal ratio 0.20、EOS loss weight 0.50を適用します。今回のvalidation悪化を踏まえ、次の50M学習ではデータ量を増やす条件も別実験として比較し、モデル容量だけの拡張と切り分けます。会話生成の改善が見えなければ、長文samplingへ戻る前に人手レビューと話題適合評価を追加します。MPSの速度またはメモリが厳しい場合は、Colab T4が復旧した時点で同じconfigを再実行できるようにします。
