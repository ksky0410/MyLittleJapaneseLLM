# 実験065：both-SFTへ一般日本語rehearsalを25%加える

## 開始前の計画

実施日は2026年9月5日、担当者はCodexです。実験064では、RealPersonaChat（RPC）とMRMPのresponse Tokenを同じ予算で混ぜたboth-SFTを行い、固定chat-test v1のToken overlap F1が0.2297となりました。一方、general validation lossは6.2432で、SFT前のbaseより悪化し、長い会話では語彙と文脈の崩れが残りました。

実験065では、064と同じboth-SFTデータへ、一般日本語Token列から作った通常next-token batchをrehearsalとして25%加えます。SFT batchのlossを75%、一般日本語rehearsalのlossを25%とし、会話の応答部分だけを学ぶ目的を保ちながら、通常文書の忘却を抑えられるか検証します。base checkpoint、Tokenizer、both-SFT train/validation、seed、学習率、学習step、EOS loss weightは064と揃え、rehearsalの有無だけを差分にします。

仮説は、rehearsal ratio 0.25によってgeneral lossと医療lossの悪化が小さくなり、both-SFTのchat F1とEOS停止率は大きく損なわれないというものです。反対に、rehearsalが会話応答の勾配を薄め、RPC・MRMPの適合や固定chat F1を下げる可能性もあります。固定48例のoverlapだけで自然さを断定せず、5 domain loss、source別の傾向、全生成本文を併せて判断します。

## 再現条件

設定は`configs/issue1-both-20m-sft-rehearsal-ratio025-mps-3k.toml`です。モデルはRoPE、LayerNorm、SwiGLU、dim 384、10層、6 heads、context length 256、19,308,032 parametersです。SFTはbatch size 8、3,000 step、AdamW、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 42、EOS loss weight 0.50です。学習率の終点は064と同じく`--lr-schedule-steps 3000`で固定します。

base checkpointは実験050の`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`、SHA-256 `326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`です。both-SFT trainは`artifacts/sft/issue1-both-balanced-v1/train.npz`、SHA-256 `645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`、response Token予算770,990です。validationは`artifacts/sft/issue1-both-full-v1/validation.npz`、SHA-256 `fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`、response Token数738,660です。rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。Tokenizerはvocab 4,096、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

実行コマンドは次のとおりです。

```bash
python3 scripts/train_sft_torch.py \
  --config configs/issue1-both-20m-sft-rehearsal-ratio025-mps-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-20m-sft-rehearsal-ratio025-mps-3k \
  --samples-dir artifacts/samples/issue1-both-20m-sft-rehearsal-ratio025-mps-3k \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.25 --lr-schedule-steps 3000 \
  --eos-loss-weight 0.5 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

元の`/Users/koseki/projects/medilink_analysis`と医師国家試験の原本は変更しません。rehearsalへ使うToken列は既存の加工済み混合列であり、一般日本語・会話・医療を含む構成と出所は既存manifestへ記録されています。重いcheckpoint本体はGitへ追加せず、metadataとSHA-256を追跡し、metrics、生成TXT、評価JSON/TXTはGitHubへ保存します。

Colab用bundleは122,165,022 bytes、SHA-256 `b19323b378fbc718619eb5becd3c91d6b7c643163d1dad3ba52f927bf448df35`です。45MB以下の3 partへ分割し、`scripts/colab_concat_065.py`で結合後のbytesとhashを検証します。bundle内の実行コード、config、base checkpoint、Tokenizer、SFT train/validation、rehearsal Token列はwrapperで個別hashも照合します。

## 成功基準

3,000 stepをNaN、OOM、shape error、checkpoint reload errorなく完走し、step 0から3,000まで100 step間隔の生成TXT、metrics、best metadataを保存することです。完走後、064と同じ5 domain、固定chat-test v1の48例、short・medium・long別F1を評価します。general・medical lossが064より改善し、EOS到達率を維持できれば仮説を支持する候補としますが、F1低下が大きい場合はrehearsal比率が強すぎると判断します。

## 実験中の記録

開始前の設定hash、学習開始時刻、step 500を超えない間隔のloss・学習率・経過時間、MPS runtime、生成本文、停止理由をこの節へ追記します。失敗、崩れた生成、評価引数の誤りも削除しません。

2026年9月5日、M3 MacBookのローカルMPSで実行を開始しようとしましたが、PyTorchが`MPSが利用できません`を返して学習開始前に終了しました。device確認の前に出力ディレクトリを作る処理へ到達していないため、065のcheckpoint、metrics、生成文はまだ作成されていません。これは現在の実行環境がheadlessでMetal deviceを利用できないためで、入力データやbase checkpointには変更がありません。Colabが復旧した場合はT4で同じbundleを実行し、復旧しない場合はCPU smokeでコードとmaskの動作だけを確認してから本学習を再試行します。

その後、`exp065-both-rehearsal025`としてColab T4割当を再試行しましたが、APIが再び`Service Unavailable`を返しました。active sessionは作成されず、bundle upload、学習、checkpoint生成は発生していません。065の本学習は未実施のままです。Colabサービスが復旧するまで、074の50M級実験など既に開始済みの処理を優先し、065は同じ固定bundleで再試行します。

本学習とは分けて、CPUの2 step smokeを`.venv/bin/python`で実行しました。最初にsystem Pythonで同じコマンドを試したところ、SentencePiece未導入のため学習開始前に終了しました。環境を`.venv/bin/python`へ固定して再実行した結果、step 1・2までNaN、shape error、checkpoint reload errorなく完走し、実測parameter数は19,308,032、学習時間は32.62秒、best validation lossは4.7213679314でした。rehearsal loss、EOS weight、lr schedule、入力hashがsummaryとmetricsへ保存され、step 0とstep 2の生成TXTも追跡対象へ追加します。これは本学習の品質結果ではなく、065のコード・mask・MPS代替環境の動作確認です。

## 結果と解釈

学習終了直後に、実測parameter数、best step、validation loss、domain loss、EOS到達率、平均生成長、Token overlap F1、代表生成、064との差、次に変える条件を追記します。短い固定promptの出力だけで会話能力を判断しません。

## 次に試すこと

065でgeneral保持と会話適合の両方が改善した場合は、ratio 0.25を暫定基準にして学習量または評価promptを増やします。会話指標が大きく下がる場合はratio 0.10またはrehearsalなしへ戻します。条件が固まったら、同じ構造を50M級へ拡大し、一般日本語・会話・医療を維持したreasoning蒸留へ進みます。
