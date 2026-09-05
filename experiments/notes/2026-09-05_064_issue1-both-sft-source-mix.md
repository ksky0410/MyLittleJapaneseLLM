# 実験064：RPCとMRMPを混ぜたboth-SFTの比較

## 開始前の計画

実施日は2026年9月5日、担当者はCodexです。Issue #1では、RealPersonaChat（RPC）とMulti-Relational Multi-Party Chat Corpus（MRMP）を標準文へ加えた条件や、両方を含めた条件を比較し、会話の助詞・語尾・相づち・短文・砕けた表現・話者交替を観察する方針です。実験062の20M事前学習ではRPCとMRMPにsource固有のvalidation改善が現れ、実験063のresponse-only SFTではRPC-SFTとMRMP-SFTで共通domainの傾向と固定chat-testの順位が変わりました。

実験064では、実験063と同じbase checkpoint、Tokenizer、context length、SFT学習率、EOS loss weight、seed、3,000 stepを使い、RPCとMRMPのresponse Tokenを半分ずつ混ぜたboth-SFTを作ります。両sourceの合計response Token数を、実験063の単独条件と同じ約770,975 Tokenへ揃えます。単純連結で学習量が増えないよう、各sourceへ半分ずつのresponse Token予算を割り当て、seed 42で決定的に抽出してから全体をshuffleします。validationはRPCとMRMPのsource別validationを全件連結し、学習中の総合validationとして使います。評価時には共通5領域と固定chat-testを両条件と同じ手順で実行します。

仮説は、both-SFTがRPC-SFTとMRMP-SFTの中間的な性質を持ち、RPCとMRMPの両方のvalidationへ改善を転移させることです。source単独条件の専門化は弱くなる可能性がありますが、短文と長めの会話の両方を学ぶことで、固定chat-testのmedium・long例や複数話者表現が安定する可能性があります。逆に、sourceを混ぜることでどちらの形式にも十分適合できず、単独条件より共通chat性能が悪化する可能性も記録します。

## 再現条件

実験開始前の基準commitは`86dd65e`です。実験064ではSFT NPZの連結用`scripts/concat_sft_npz.py`、sourceごとのresponse Token予算を均等にした混合用`scripts/mix_sft_npz.py`、both-SFT用configを追加します。これらをテストしてcommit・pushしてから、データ作成と学習を開始します。

base checkpointは実験050の`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`で、SHA-256は`326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`です。Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。モデルはRoPE、LayerNorm、SwiGLU、dim 384、10層、6 heads、context length 256、19,308,032 parameterです。学習はbatch size 8、3,000 step、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、EOS loss weight 0.50、seed 42、学習率schedule終点3,000 stepで実行します。

学習データは`artifacts/sft/issue1-both-balanced-v1/train.npz`です。RPCとMRMPを各385,487〜385,488 response Tokenへ割り当て、合計約770,975 Tokenとします。validationは`artifacts/sft/issue1-both-full-v1/validation.npz`で、RPCとMRMPのsource別validation全件を連結します。SFTの学習対象はloss maskが1のresponse本文と末尾EOSだけです。元の`artifacts/corpus/conversation-v1`と、元データを置く`/Users/koseki/projects/medilink_analysis`は変更しません。

実行コマンドは次のとおりです。

```bash
python scripts/train_sft_torch.py \
  --config configs/issue1-both-20m-sft-source-colab-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-20m-sft-source-colab-3k \
  --samples-dir artifacts/samples/issue1-both-20m-sft-source-colab-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 --device mps
```

Colab T4も利用可能なら候補にしますが、実験063でT4が3回連続503、L4がquota拒否となったため、まずはMPSで確実に実行します。学習途中のstep 0から3,000まで100 step間隔の生成、metrics、checkpoint metadataを保存し、重いcheckpoint本体はGit管理外としつつhashを記録します。

## 成功条件

both-SFTが3,000 stepまでNaN、OOM、shape errorなく完走し、RPC-SFT・MRMP-SFTと同じbase、response Token予算、seed、学習条件で比較できることです。共通5領域、固定chat-test 48例、stepごとの全文生成、best checkpoint metadataを保存します。source単独条件より良くならない場合も、混合による専門化と汎化のトレードオフとして記録します。

## 実験中の記録

データ入力hash、source別選択数、response Token数、base hash、コードcommit、MPS runtime、学習のstep・loss・学習率・経過時間、異常、生成本文の回収状況を節目ごとに追記します。混合後の低品質な生成や早期EOSも削除しません。

## 実験終了後の結果と解釈

ここへboth-SFTのbest step、validation loss、共通5領域のloss、固定chat-testのEOS・生成長・Token overlap、RPC-SFT・MRMP-SFTとの差、代表的な生成を追記します。bothの総合validationが低くても、source別の改善を失っていないかを分けて確認します。F1とEOSは短い定型応答の影響を受けるため、自然な会話能力の証拠とは断定しません。

## 次に試すこと

both-SFTでsource固有の改善が保たれるなら、次はboth-SFTへrehearsal lossを加えてgeneral lossの悪化を抑えます。混合で性能が落ちるなら、RPCとMRMPの配分、source別sampling、またはSFT前のpretraining量を一つずつ変えて再検証します。
