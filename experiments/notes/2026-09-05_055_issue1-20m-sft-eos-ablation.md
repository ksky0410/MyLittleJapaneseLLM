# 実験055：20M SFTのresponse末尾EOS loss ablation

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。Issue #1の会話SFTを進めた049〜054では、話者境界を保ったresponse-only maskとrehearsalを使いました。054ではconversation形式のsample promptにより形式上の早期終了は改善しましたが、短い会話応答が挨拶へ寄りやすく、EOSの学習が強すぎる可能性が残っています。

055では、実験054と同じ20M base、ratio 0.50、会話NPZ、rehearsal Token列、seed、optimizer、学習率、3,000 stepを使い、response末尾EOSをSFT lossの対象から除く条件を追加します。054のEOS込み条件を比較対象として再利用し、変更をEOS maskだけに限定します。

仮説は、EOSを学習しない条件では短すぎるcompletionが減り、実在話者IDと履歴を使った生成の内容が続きやすくなるというものです。一方、EOSを除くと応答終了を学習できないため、長すぎる生成や会話境界の無視が増える可能性もあります。validation lossだけでなく、EOS数、平均生成長、固定48問、生成全文、5 domain loss、人手レビューを比較します。

## 使用条件と再現方法

モデルは19,308,032 parameters、dim 384、10層、6 heads、context 256、RoPE、LayerNorm、SwiGLUです。初期checkpointは実験050の`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`、SHA-256 `326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`です。設定は`configs/issue1-both-20m-rehearsal-ratio050-noeos-colab-3k.toml`で、学習中sampleは`DA`と`DC`の実在話者IDを使うconversation形式です。

新しいCLI引数`--exclude-eos-from-sft-loss`を`scripts/train_sft_torch.py`へ追加し、mask処理のテストを追加します。Colab wrapperは`scripts/colab_bootstrap_055.py`、bundle連結は`scripts/colab_concat_055.py`、archive化は`scripts/colab_package_055.py`です。学習開始前にコード、config、bundle hash、連結scriptをcommitしてpushします。

学習前の基準commitは`426e884`です。configのSHA-256は`b7e3ab2a69559ede23c9d37680215c7d424d64084be854608f626f9189803cf8`、更新後の`train_sft_torch.py`は`76f61c4a56b63177bfc9387735460b958145d0a7da91ee31bda7c7de0e30a01d`です。`/tmp/exp055_bundle.tar.gz`は271,721,152 bytes、SHA-256 `682c8f778e5af94b7ce7cedf81d7bf8a9b735225a62d704a216b25cb4bbc0816`です。45MBのpart 5個と、残りを17MB、17MB、137KBへ分割した3個を使います。連結scriptへこのbytesとhashを固定しました。更新後の`colab_concat_055.py`のSHA-256は`257137bff7ce16c6ca9ba8b3dab68139e55afa18e24be08a4b091bdbeddb2cea`です。この状態をpushしてからuploadを開始します。

## 実行コマンド

```bash
uv run pytest -q
tar -czf /tmp/exp055_bundle.tar.gz \
  configs/issue1-both-20m-rehearsal-ratio050-noeos-colab-3k.toml \
  scripts/train_sft_torch.py scripts/train_torch.py scripts/_common.py \
  src artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.json \
  artifacts/sft/chat-v1-context256/train.npz \
  artifacts/sft/chat-v1-context256/validation.npz \
  artifacts/tokens/mixed-ja-80-10-10-v2-train.bin
split -b 45m -d -a 2 /tmp/exp055_bundle.tar.gz /tmp/exp055_bundle_part_
split -b 17m -d -a 1 /tmp/exp055_bundle_part_05 /tmp/exp055_last_part_
colab new --session exp055-20m-noeos-sft --gpu T4
colab exec --session exp055-20m-noeos-sft --timeout 120 --file scripts/colab_concat_055.py
colab exec --session exp055-20m-noeos-sft --timeout 3600 --file scripts/colab_bootstrap_055.py
colab exec --session exp055-20m-noeos-sft --timeout 120 --file scripts/colab_package_055.py
```

bundle連結時にbytesとSHA-256を検証し、学習後は軽量archive、manifest、best checkpointを回収してsessionを停止します。重い`.pt`はGitへ追加せず、metadataとノートにhashを残します。生成TXT、metrics、評価JSON/TXTはGitの追跡対象です。

## 成功条件

EOS除外条件が3,000 stepまで完走し、NaN、OOM、shape error、mask対象不足、checkpoint reload errorがないことです。step 0から3,000まで100 step間隔の生成TXT、500 step間隔のmetadata、5 domain評価、固定chat 48例を保存します。

## 実施記録

15:35 JST、新規session `exp055-20m-noeos-sft`へT4を割り当て、分割した8個のpartをuploadしました。連結後のbundleは271,721,152 bytes、SHA-256 `682c8f778e5af94b7ce7cedf81d7bf8a9b735225a62d704a216b25cb4bbc0816`で、開始前に固定した値と一致しました。wrapperの入力hashもconfig、EOS除外を含むSFT script、base checkpoint、train/validation NPZ、rehearsal Token列、Tokenizerのすべてで一致しました。

EOS除外条件は3,000 stepまでNaN、OOM、shape error、mask対象不足、checkpoint reload errorなしで完走しました。Colab環境はPyTorch 2.11.0+cu128、CUDA 12.8、Tesla T4、AMP有効でした。実測parameter数は19,308,032、peak allocatedは750,051,840 bytes、peak reservedは817,889,280 bytes、経過時間は248.03秒でした。bestはstep 2,400で、SFT validation loss 3.939761、PPL 51.4063でした。best checkpointは`artifacts/checkpoints/issue1-both-20m-rehearsal-ratio050-noeos-colab-3k/best.pt`、SHA-256 `2b894d75837381a8cc886dbcab7c4249ecc40ac6a7de1a43649650cb4cab4fbf`です。軽量archiveは40ファイル、7,084 bytes、SHA-256 `8997814168a1ec87d8c1355568154aa3bcf9463ae8253ea051b5518fccd58ecf`でした。archive、manifest、best weightを回収し、学習後にColab sessionを停止して`colab sessions`が空であることを確認しました。

学習中サンプルは054と同じ実在話者ID `DA`・`DC`のconversation形式です。step 0では054と同じ崩れた続きを出しましたが、step 1,000では「こんにちは!よろしくお願いします。お願いします!!あなたは?」、step 2,000では「こんばんは!今日もお願いいたします。」、step 3,000では「こんにちは!よろしくお願いします!よろしくお願いします!」となりました。EOSをloss対象から除いたため応答が早く終了しなくなった一方、挨拶の反復と過生成が現れました。step 0から3,000まで100 step間隔の全31個の生成TXTを保存しています。

best checkpointをローカルPyTorch 2.14.0 CPUでreloadし、052・054と同じ5 domainを20 batchずつ評価しました。general lossは5.1779（PPL 177.30）、conversationは2.8318（PPL 16.98）、medicalは3.0903（PPL 21.98）、RealPersonaChatは2.8250（PPL 16.86）、MRMPは2.3748（PPL 10.75）でした。054のEOS込み・ratio 0.50・3,000 stepの5.1347、2.8062、3.0669、2.7972、2.3432より、5領域すべてで悪化しました。評価JSONは`artifacts/evaluations/issue1-both-20m-rehearsal-ratio050-noeos-colab-3k-domains.json`です。

固定chat-test v1の48例では、EOS 42/48、平均生成長34.02 Token、precision 0.1331、recall 0.2841、Token overlap F1 0.1596でした。short・medium・longのF1は0.2004、0.1195、0.1589でした。054のEOS込み条件はEOS 48/48、平均10.06 Token、F1 0.2080でした。EOS除外によって生成長とrecallは増えましたが、precisionとF1が下がり、固定会話では適切な終了と内容適合を両立できませんでした。評価JSON/TXTは`artifacts/evaluations/issue1-both-20m-rehearsal-ratio050-noeos-colab-3k-chat-test-v1.json`・`.txt`、レビュー用JSONは`experiments/evaluation/issue1-both-20m-rehearsal-ratio050-noeos-colab-3k-chat-review.json`です。

054と055を比較すると、EOSを除く仮説は「短すぎるcompletionを減らす」という点では支持されましたが、validation、5 domain、固定chat F1、生成の反復では支持されませんでした。現時点では、EOS込みで学習した054をSFTの基準条件として維持します。EOSを単純に除くのではなく、response長や会話境界を考慮したEOSの重み付けを次に試す価値があります。ただし、この結果だけでEOS重みの最適値を決めず、まず複数の実在履歴を使った人手レビューで反復と自然さを確認します。

## 結果の要約

055はEOS除外条件で完走しましたが、SFT validation、5 domain、固定48問のToken overlap F1では054のEOS込み条件を下回りました。生成長とrecallは増えた一方、EOS停止、precision、挨拶の反復、過生成が悪化したため、次のSFT基準はEOS込みの054へ戻します。単純なEOS除外ではなく、応答長や会話境界を考慮した重み付けを別実験として検討します。

## 次に試すこと

EOS除外で長さと自然さが改善すれば、EOSを別重みで学習する中間条件や、response長別のloss weightingを試します。改善しなければ、EOS込みratio 0.50を基準として50M級モデルやLoRA/QLoRAへ進みます。
