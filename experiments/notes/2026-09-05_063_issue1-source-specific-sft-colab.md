# 実験063：RPC単独・MRMP単独の応答部分SFTを比較

## 開始前の計画

実施日は2026年9月5日、担当者はCodexです。実験062では、20M級モデルを同じ学習量で事前学習しても、RPC条件はRPC validation、MRMP条件はMRMP validationへ強く適合することが分かりました。一方、固定chat-testでは両条件とも早期EOSが多く、source適合性の差が自然な応答へどの程度つながるかは判定できませんでした。実験063では、Issue #1の会話データを履歴と応答へ分け、response-only SFTでRPCとMRMPの性質を比較します。

今回は、元の`artifacts/corpus/conversation-v1`を変更せず、`dataset`フィールドでRPCとMRMPを抽出した派生JSONLを作ります。SFTのtrainデータは、各sourceの全例をいったん同じTokenizerとcontext length 256でNPZ化し、その後、response部分のToken数が少ないsourceに合わせて両条件を同じresponse Token予算へ削減します。抽出順はseed 42で固定し、validationはsource別の全例を使います。これにより、RPCの会話数が多いことをそのまま学習量の差にしません。

仮説は、RPC条件はRPC validationと1対1の話題継続に、MRMP条件はMRMP validationと複数話者の短い応答に強くなることです。事前学習だけで見えたsource固有差がSFTでも再現される可能性がありますが、応答部分だけを学ぶことで固定chat-testの早期EOSが改善する可能性もあります。自動Token overlapは参照文との表面的な一致にすぎないため、5領域loss、EOS、生成長、全文生成を一緒に保存します。

## 再現条件

実験開始前の基準commitは`fe1f96f`です。実験063では、source抽出用`scripts/filter_conversation_sources.py`と、response Token予算を固定する`scripts/select_sft_npz.py`を追加します。実行コードと準備記録をcommit・pushしてから、データ生成と学習へ進みます。

モデルは実験056の20M級基盤モデルを初期値にします。現在ローカルにあるbase checkpointは`artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k/best.pt`で、SHA-256は`f554bb5b6b2b1fe20b9318d1558465c0bb4407ddfa971593b853dc7f2aab868e`です。モデルはvocab 4,096、dim 384、10層、6 heads、context length 256、RoPE、LayerNorm、SwiGLU、19,308,032 parameterです。SFTはbatch size 8、3,000 step、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 42、EOS loss weight 0.50、学習率scheduleの終点3,000 stepで実行します。RPCとMRMPを同じColab T4 runtime上で順番に実行します。

train NPZのresponse Token予算は、source別full preparation後に少ない側の総response Token数を採用します。RPCはこの予算までseed 42の決定的なsubsetを抽出し、MRMPも同じ予算で抽出します。抽出後のexample数とresponse Token数、NPZ hashはmanifestへ保存します。validationはRPC 1,365会話、MRMP 89会話のsource別JSONLから作成し、評価時にはgeneral、conversation、medical、RPC、MRMPの同じ5領域を使います。

source抽出後のtrain会話数はRPC 10,851、MRMP 784、validation会話数はRPC 1,365、MRMP 89でした。full SFT preparationではRPC trainが315,584 examples・response 5,132,071 Token、MRMP trainが81,382 examples・response 770,975 Tokenとなりました。MRMPのresponse Token数を共通予算として採用し、RPCからseed 42で47,315 examples・770,979 response Tokenを選び、MRMPは81,382 examples・770,975 response Tokenを選びました。選択後NPZのSHA-256はRPC `476b497eaf13653d894f82c78bb194c928daaaa7f2aef119967e1f59a6f71cc7`、MRMP `bce88dc2c7ef83093923c0e166e485360bafc6177b6be99813664610169cb7af`です。

SFTの実行コマンドは、準備したsource別NPZに対して次の形式で実行します。

```bash
python scripts/train_sft_torch.py \
  --config configs/issue1-rpc-20m-sft-source-colab-3k.toml \
  --base-checkpoint artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k/best.pt \
  --train-data artifacts/sft/issue1-rpc-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-rpc-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-rpc-20m-sft-source-colab-3k \
  --samples-dir artifacts/samples/issue1-rpc-20m-sft-source-colab-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 --device auto
```

MRMPではconfig、train-data、validation-data、output-dir、samples-dirだけを対応する名前へ置き換え、他の条件は固定します。Colab成果物はstep 0から3,000まで100 step間隔の生成本文、metrics、summary、checkpoint metadataを回収します。重いcheckpoint本体はGit管理外とし、best checkpointのhashをmanifestとノートに保存します。

## 成功条件

RPCとMRMPが同じbase checkpoint、同じresponse Token予算、同じSFT条件、同じseedで3,000 stepまで完走し、NaN、OOM、shape errorが発生しないことです。各条件のsource別validation、固定chat-test 48例、学習途中の全生成文、最良checkpointのhashを保存します。source別loss差が小さい場合も、SFTで差が検出できなかった結果として記録します。

## 実験中の記録

元JSONLのhash、source抽出件数、full NPZの件数、response Token予算、選択後NPZのhash、base checkpointのhash、bundleとColab session、各条件の開始・終了、runtime、異常、生成回収状況を節目ごとに追記します。生成文は良し悪しに関係なく、GitHubで細かく確認できる形で保存します。

source抽出とfull SFT preparationは2026年9月5日に完了しました。元の`artifacts/corpus/conversation-v1`は読み取りのみで扱い、RPC/MRMPの派生JSONL、full NPZ、balanced NPZを新しい出力先へ作成しました。元JSONLのtrain hashは`be28c0b9213de7e2ad8b3dcdf3f687369dd3bb2aead38213acb4885a6d75c6e7`、validation hashは`bac9710f0e1ebbbc955debb7e6a773c450abd63f71869f97bd0da73740d8cdde`、test hashは`65f534a8e63acf056bcbcbc7c827d62ff7dedfd383be382cc299c056dec90ce5`です。RPC source manifestのSHA-256は`291226b76aaa8faed6baa3d6ea95079219a4811c760bc244a2d8239401b6d305`、MRMP source manifestは`a14443c14a7c19f8c0788cbfc918091a16ac4383520bf631ffbb0cd44e8a181b`です。full SFT manifestはRPC `663855c29f6952d27bb2c3721ffdfdb6c529b7080e071d75866d821480df5b6b`、MRMP `0e98e9f692e787b3d421b450240aba1be8892aa31504554e56d5d1c7d836eaec`、balanced selection manifestはRPC `53bf8efab5c9851c0e79c241047af4c623eff69ae06325cc63d0aed818cf3ffc`、MRMP `11483fd0e51a537545ec5bbe11e4c41ff5e3dffaa0a26f7f0184dbd194569c4b`です。

実行コードと設定はcommit `c6bab65`（`exp: prepare 063 source-specific sft`）としてpush済みです。Colab用bundleは`/tmp/small_llm-colab-063-c6bab65.tar.gz`、サイズ約144MB、SHA-256 `4d34971c7ac3c4fad1640888484e82edb51b4e255a5722a478c4aba775d6ea0f`です。bundleにはSFT用の4つのNPZ、base checkpoint本体とmetadata、Tokenizer、モデル実装、学習script、2つのconfigを含め、元のJSONLは含めていません。

Colab session `exp063-source-sft`の新規T4割当は、2026年9月5日にHTTP 503 `Service Unavailable`で失敗しました。upload、bundle展開、学習、出力変更は発生していません。失敗直後に`colab sessions`を確認し、active sessionがないことを確認しました。Colabサービスの一時的な割当失敗として残し、再試行します。

`exp063-source-sft-retry`としてT4割当を再試行しましたが、同じHTTP 503 `Service Unavailable`で失敗しました。2回ともactive sessionは作られておらず、bundle uploadと学習は未実施です。T4側の一時障害か割当制限かを切り分けるため、対応GPUのL4を一度試します。

L4も試しましたが、Colab CLIから「accountのquotaまたはentitlementがない」と拒否されました。L4 sessionは作成されていません。現時点でColabにactive sessionはなく、T4の一時的な503が解消するかを最後に再確認します。

その後のT4最終再試行もHTTP 503 `Service Unavailable`で失敗し、Colab sessionは作成されませんでした。T4 3回、L4 1回の割当失敗ではbundle upload、学習、既存成果物の上書きは発生していません。ローカルではPyTorch 2.14.0、MPS build有効、`torch.backends.mps.is_available()`が`True`であることを確認したため、実験条件を揃えたままM3 MacBookのMPSへ切り替えます。Colab T4での実行ができなかった事実は失敗記録として残し、RPCとMRMPは同じローカルMPS runtimeで順番に実行します。

MPSでの初回起動は、既存`train_sft_torch.py`が`mps`をdeviceとして受け付けず、学習前の引数検証で終了しました。checkpointや生成物は作られていません。SFT scriptへMPSの可用性検証と、`auto`時にCUDAがなければMPSを選ぶ処理を追加し、parserテストを含む全85テストが通過しました。この修正commitは実行前に追記します。

## 実験終了後の結果と解釈

ここへデータ件数、SFTの実測parameter数、runtime、学習時間、best step、source別loss、固定chat-testのEOS・生成長・Token overlap、代表生成の観察を追記します。SFT validation lossは応答マスク部分の次Token予測であり、一般知識、医学的正確性、安全性、会話の人間らしさを直接保証しません。source別のvalidationが良くても、抽出元への過適合や定型表現の記憶を切り分けます。

## 次に試すこと

結果に応じて、両sourceを同時に含むSFT、SFTとrehearsalの混合、EOS loss weightの再比較、または固定promptを現代口語へ広げた人手レビューへ進みます。一度に複数の変更を入れず、source配分、学習対象、rehearsal、EOS制御を分けて検証します。
