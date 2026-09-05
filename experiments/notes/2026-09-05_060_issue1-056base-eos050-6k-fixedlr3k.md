# 実験060：6,000 step学習でcosine学習率を3,000 stepに固定する

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。Issue [#1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)の方針に沿い、一般日本語を維持した056の日本語基盤へ会話SFTを続けます。実験059ではEOS weight 0.50、rehearsal ratio 0.50で6,000 stepを実行し、3,000 step条件よりchat F1とdomain lossが改善しましたが、`max_steps=6000`によってcosine学習率の曲線も変わっていました。

059の改善が追加学習stepによるものか、学習率曲線によるものかを分けるため、今回は6,000 stepまで学習しながら、`--lr-schedule-steps 3000`で学習率を3,000 step時点で下限へ到達させます。059と同じbase checkpoint、データ、seed、batch、EOS weight、rehearsal ratioを使い、学習率の終点だけを3,000 stepへ戻します。仮説は、改善が追加stepの効果なら、060でも059に近いF1とdomain lossが得られるというものです。改善が主に高い学習率の継続によるものなら、060は057の3,000 step条件に近づくと予想します。

## 条件と再現情報

059用config `configs/issue1-056base-rehearsal-ratio050-eos050-colab-6k.toml`を使用し、実行コードの新機能`--lr-schedule-steps`で学習率曲線だけを制御します。モデルはRoPE・LayerNorm・SwiGLU、19,308,032 parameters、batch size 8、context length 256、max steps 6,000、lr schedule steps 3,000、EOS loss weight 0.50、rehearsal ratio 0.50、seed 42です。入力は059と同じbase checkpoint、会話train/validation、rehearsal Token列、Tokenizerを使います。元の医師国家試験データと`/Users/koseki/projects/medilink_analysis`は変更しません。

実行前に実験wrapper、学習率分離機能、bundle検証、package、noteをcommit・pushします。重いcheckpoint本体はGitへ追加せず、入力hashとColab manifestのcheckpoint hashを残します。生成文はstep 0から6,000まで100 step間隔で保存し、すべてGitHubへ保存します。

## 成功基準

6,000 stepをNaN、OOM、shape errorなく完走し、metrics、summary、checkpoint metadata、生成TXT、domain評価、固定48例chat評価を回収できれば実装上の成功とします。059と同じ評価条件で、EOS到達率、平均生成長、token overlap F1、short・medium・long別F1、general・conversation・medical・RPC・MRMP lossを比較します。学習率の記録がstep 3,000以降で最小学習率に張り付いていることも確認します。

## 実験中の記録

bundle hash検証、Colab割当、学習開始・途中・完了、評価、回収、session停止を時系列で追記します。失敗、停止、生成崩れも削除せずに残します。

## 結果と解釈

実験終了直後に、EOS 0.50・6,000 step・schedule 3,000 stepのloss、PPL、runtime、生成例を追記します。059との差から、追加学習stepと学習率曲線のどちらが主な要因かを解釈します。

## 次に試すこと

060で059の改善が再現されれば、EOS 0.50とrehearsal ratio 0.50を暫定標準候補にします。再現されなければ、059の結果は学習率曲線の寄与が大きいと判断し、学習率とSFT stepを別々に探索します。その後、rehearsal ratioを0.25または0.75へ一つずつ変え、条件が固まった段階で20M構造を50Mへ拡大し、reasoning蒸留へ進みます。
