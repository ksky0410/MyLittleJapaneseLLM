# 実験052：20M rehearsal ratio sweep

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験051では、20Mの`both` checkpointから会話応答SFTを行ったところ、rehearsal ratio 0.25がSFT-onlyよりgeneral、medical、conversation、RPC、MRMPの全domain lossで低く、固定chatのToken overlap F1もわずかに高くなりました。ただし、ratio 0.25が適量かは未検証です。052では、同じ20M base checkpoint、同じ会話NPZ、同じrehearsal Token列、同じoptimizer・学習率・seed・1,000 stepでratio 0.10と0.50を追加し、051の0.25と比較します。

比較条件はrehearsal ratio 0.10、0.25、0.50です。ratio 0.25は実験051で完了した成果物を再利用し、052では0.10と0.50だけを実行します。ratioが高いほどgeneral・medicalの保持は良くなる一方、応答maskへ割り当てられるbatchが減り、会話応答の適合が遅くなると予想します。ratio 0.10はSFT寄り、0.50は保持寄りの条件として比較します。

仮説は、0.25または0.50が通常domain lossと会話domain lossのバランスで良く、0.10は固定chatの応答適合で良いというものです。固定chatのoverlap、EOS、生成長だけでは自然さを判定せず、5領域の通常loss、SFT validation loss、生成全文、人手レビュー用JSONを保存します。比率そのもの以外を変えないため、既存0.25結果との比較時には、同じ実験051のbase checkpointとconfigを使ったことを明記します。

## データ、モデル、再現条件

初期値は実験050の`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`、SHA-256 `326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`です。モデルは19,308,032 parameters、dim 384、10層、6 heads、context 256、RoPE、LayerNorm、SwiGLUです。会話train NPZは`artifacts/sft/chat-v1-context256/train.npz`、validation NPZは同ディレクトリの`validation.npz`、rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`です。これらのSHA-256は051の記録を参照し、元のJSONL、医師国家試験データ、`/Users/koseki/projects/medilink_analysis`は変更しません。

設定は実験051と同じ`configs/issue1-both-20m-sft-torch-colab-1k.toml`、SHA-256 `e2b01afa98a28ea7b863a7b1ffe02e86088cd73009fc3e5422a241fd2c3a177b`です。実験開始時点の基準commitは051完了後の`c3b9574`です。Colab用wrapperは`scripts/colab_bootstrap_052.py`、成果物packageは`scripts/colab_package_052.py`、分割bundle連結とhash検証は`scripts/colab_concat_052.py`です。各SHA-256はwrapper `8f7d8bf4bbf6b6f33cf1f02ffd213671c6ae7efe4921ec2869122f28d8e58f31`、package `c16fe6caf80d97659fdac79c7f20885e1c7d90d5667dbf628c0055adbc72e63b`、concat確定版 `77d996b28151a6c0d3777b300efd2c8cc586d77bf123b9d18a3ca4c98b49bfa1`です。052は051と同一のbundle（259MB、SHA-256 `94f28a741d6e3bebf922031ed8feafa1ecf2eaaacba54da8c38ab1e2950cbd35`）を使います。

## 実行コマンド

```bash
colab new --session exp052-20m-ratio --gpu T4
colab upload --session exp052-20m-ratio /tmp/exp052_bundle_part_00 /content/exp052_bundle_part_00
colab upload --session exp052-20m-ratio /tmp/exp052_bundle_part_01 /content/exp052_bundle_part_01
colab upload --session exp052-20m-ratio /tmp/exp052_bundle_part_02 /content/exp052_bundle_part_02
colab upload --session exp052-20m-ratio /tmp/exp052_bundle_part_03 /content/exp052_bundle_part_03
colab upload --session exp052-20m-ratio /tmp/exp052_bundle_part_04 /content/exp052_bundle_part_04
colab upload --session exp052-20m-ratio /tmp/exp052_last_part_0 /content/exp052_last_part_0
colab upload --session exp052-20m-ratio /tmp/exp052_last_part_1 /content/exp052_last_part_1
colab upload --session exp052-20m-ratio /tmp/exp052_last_part_2 /content/exp052_last_part_2
colab exec --session exp052-20m-ratio --timeout 120 --file scripts/colab_concat_052.py
colab exec --session exp052-20m-ratio --timeout 1800 --file scripts/colab_bootstrap_052.py
colab exec --session exp052-20m-ratio --timeout 120 --file scripts/colab_package_052.py
```

各条件の実行後に軽量archive、manifest、best checkpointを回収し、sessionを停止します。T4割当、bundle連結、学習途中停止、回収失敗も削除せず記録します。

## 成功条件

ratio 0.10と0.50が同じ初期値から1,000 stepまで完走し、NaN、OOM、shape error、mask対象不足、checkpoint reloadエラーがないことです。実験051のratio 0.25と合わせて、SFT validation、general・conversation・medical・RPC・MRMPのdomain loss、固定chat 48例、生成TXT、metrics、summary、checkpoint metadataを比較できることです。

## 実験中の記録

開始前、Colab割当、bundle hash検証、各ratioの開始・途中・終了、回収、評価をこのノートへ追記します。生成された文章は品質に関係なく全step保存し、空出力や特殊Token混入も削除しません。

15:00 JST、新規session `exp052-20m-ratio`へT4を割り当て、051と同じbundleを使用して実行を開始しました。通常の259MB一括uploadは使わず、既着の5個の45MB partと、転送が停滞した最後のpartを17MB、17MB、136KBへ分割した3個のpartを連結しました。連結結果は271,720,679 bytesで、予定SHA-256 `94f28a741d6e3bebf922031ed8feafa1ecf2eaaacba54da8c38ab1e2950cbd35`と一致しました。最後のpartの転送停滞と再分割は失敗として隠さず記録し、bundleの内容変更はありません。ratio 0.10を先に実行し、その後ratio 0.50を同じT4上で実行しました。

両条件とも1,000 stepまで完走し、NaN、OOM、shape error、mask対象不足、checkpoint reloadエラーは発生しませんでした。実測parameter数は各19,308,032です。Colab環境はPyTorch 2.11.0+cu128、CUDA 12.8、Tesla T4、AMP有効でした。ratio 0.10は経過89.68秒、peak allocated 764,613,632 bytes、peak reserved 843,055,104 bytesで、step 1,000がbestでした。bestのtrain lossは4.176839、SFT train lossは4.147194、rehearsal train lossは4.443645、SFT validation lossは3.902511、PPLは49.5266でした。ratio 0.50は経過86.42秒、peak allocated 750,051,840 bytes、peak reserved 817,889,280 bytesで、step 1,000がbestでした。bestのtrain lossは3.606019、SFT train lossは3.836696、rehearsal train lossは3.375341、SFT validation lossは3.923542、PPLは50.5793でした。

ratio 0.10のbest weightは`artifacts/checkpoints/issue1-both-20m-rehearsal-ratio010-colab-1k/best.pt`、SHA-256 `3b097acdceebdc25c3c42666e611b391571b3111fb244e24d0eca03415f6dc23`です。ratio 0.50は`artifacts/checkpoints/issue1-both-20m-rehearsal-ratio050-colab-1k/best.pt`、SHA-256 `9920c1a78c1665c6ff93cc293b101824f565c1d582ea5fa66f0dd8f45c801b21`です。Colabの軽量archiveは32ファイル、5,909 bytes、SHA-256 `3fc58c4a8600405a4edbc6ebdb1a8150c9ce61227bb7918c0e63ec7b34c8d7fc`でした。metrics、summary、step 500/1,000のmetadata、step 0から1,000まで100 step間隔の生成TXTを回収し、学習終了後にsessionを停止して`colab sessions`が空であることを確認しました。

学習中の固定生成は両条件で同じ挙動でした。step 0では「今日なにしてた？」に続いて長いものの崩れた文章が出ましたが、step 500以降は質問をそのまま繰り返した直後に終了する短い出力へ変化しました。この悪化した出力も含め、全22個の生成TXTを削除せず保存しています。自由生成の一例だけではSFT条件の優劣を決められないため、固定chat評価とdomain lossを併用します。

回収後、ローカルPyTorch 2.14.0 CPUで同じbest checkpointをreloadし、general、conversation、medical、RealPersonaChat、MRMPの各validationを20 batchずつ評価しました。ratio 0.10は順にloss 5.8176、3.0274、3.3210、2.9891、2.5751（PPL 336.16、20.64、27.69、19.87、13.13）でした。ratio 0.50はloss 5.4677、2.9771、3.1897、2.9533、2.4980（PPL 236.91、19.63、24.28、19.17、12.16）でした。ratio 0.50は5領域すべてでratio 0.10より低いlossとなり、一般・医療・会話の通常Token評価では0.50が優勢でした。評価結果は`artifacts/evaluations/issue1-both-20m-rehearsal-ratio010-colab-1k-domains.json`と`artifacts/evaluations/issue1-both-20m-rehearsal-ratio050-colab-1k-domains.json`へ保存しました。

固定chat-test v1は同じ48例、`max_new_tokens=64`、temperature 0.8、top-k 40、seed 42で評価しました。ratio 0.10はEOS 48/48、平均生成長9.60 Token、Token overlap precision 0.2658、recall 0.1870、F1 0.1849でした。ratio 0.50はEOS 48/48、平均生成長9.42 Token、precision 0.2776、recall 0.2070、F1 0.2020でした。ratio 0.50は全体F1で0.0171高く、short・medium・longのF1もそれぞれ0.3210、0.1478、0.1372となり、ratio 0.10の0.2790、0.1410、0.1348をすべて上回りました。ただしF1は参照応答とのToken重複に過ぎず、生成の自然さ・妥当性・医学的正確性を保証しません。生成本文を含むJSON/TXTは`artifacts/evaluations/issue1-both-20m-rehearsal-ratio010-colab-1k-chat-test-v1.json`・`.txt`とratio 0.50の同名ファイルへ保存し、人手レビュー用JSONは`experiments/evaluation/issue1-both-20m-rehearsal-ratio010-colab-1k-chat-review.json`と`experiments/evaluation/issue1-both-20m-rehearsal-ratio050-colab-1k-chat-review.json`へ保存しました。

051のratio 0.25はSFT validation loss 3.9058、general 5.6421、conversation 2.9997、medical 3.2562、RealPersonaChat 2.9630、MRMP 2.5305、固定chat F1 0.1916でした。今回の3条件を並べると、SFT validationだけなら0.10（3.9025）が最も低く、通常5領域lossと固定chat F1では0.50が最も低くなりました。ratio 0.25はgeneral・conversation・medical・RealPersonaChat・MRMPの通常domain lossで0.50よりわずかに高く、固定chat F1では0.50より低い結果でした。したがって、この20M・1,000 stepの条件ではratio 0.50を次の長時間SFTの第一候補とします。ただし、0.10との差は小さく、自由生成がstep 500以降に短縮したことも含め、最適比率とは断定しません。学習量、評価データとの重複、Token overlapの限界を考慮し、人手レビューと長い学習で再確認します。

## 結果と解釈

ratio 0.10、0.25、0.50について、SFT validationと通常domain lossのトレードオフ、固定chatのEOS・生成長・overlap、生成本文の人手レビューを分けて記録します。小規模な20Mモデルと1,000 stepの結果であり、一般的な最適ratioや会話性能を断定しません。

## 次に試すこと

ratioの傾向が見えた場合は、選んだratioで学習Token量を増やします。差が小さい場合は短い応答の層化sampling、source別SFT、または日本語instruction dataの蒸留へ進みます。安定したSFT条件を確認した後、LoRA/QLoRAやモデルサイズ拡大と比較します。
