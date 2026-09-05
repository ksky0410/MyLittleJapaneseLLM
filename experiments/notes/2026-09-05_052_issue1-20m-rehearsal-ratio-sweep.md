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

## 結果と解釈

ratio 0.10、0.25、0.50について、SFT validationと通常domain lossのトレードオフ、固定chatのEOS・生成長・overlap、生成本文の人手レビューを分けて記録します。小規模な20Mモデルと1,000 stepの結果であり、一般的な最適ratioや会話性能を断定しません。

## 次に試すこと

ratioの傾向が見えた場合は、選んだratioで学習Token量を増やします。差が小さい場合は短い応答の層化sampling、source別SFT、または日本語instruction dataの蒸留へ進みます。安定したSFT条件を確認した後、LoRA/QLoRAやモデルサイズ拡大と比較します。
