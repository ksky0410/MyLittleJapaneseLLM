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

2026-09-05 17時台に、059と同じ入力を使い、学習率分離機能を含むbundleを271,827,848 bytes、SHA-256 `44d2f1cad370dd346af6f68b05a67ab2c70d75617b90bd5583b1341fa0f43e07`として作成しました。Colab上で6分割bundleを再結合し、サイズとSHA-256が一致しました。学習率分離機能のコミットは`82ae769`、060の実験wrapperと計画のコミットは`901056c`です。

ColabのPythonは3.13.15、PyTorchは2.11.0+cu128、CUDAは12.8、GPUはTesla T4でした。`lr_schedule_steps=3000`を含むhash検証を通過し、6,000 stepをNaN、OOM、shape errorなしで完走しました。学習時間は494.344925秒、最大allocated memoryは750,051,840 bytes（約715 MiB）でした。

学習完了後、059と同じ評価用の一般・会話・医療・RealPersonaChat・MRMP validationと固定48例のchat-test-v1を同じColab T4で評価しました。評価wrapperのコミットは`d781411`です。評価と成果物回収後にColab sessionを停止し、`colab sessions`でアクティブなsessionがないことを確認しました。

## 結果と解釈

best checkpointはstep 5,800で、validation lossは3.2746284962、perplexityは26.4334035456でした。final step 6,000ではvalidation lossが3.2785223603へわずかに悪化しました。best時点のtrain loss、SFT部分のloss、rehearsal部分のlossは、それぞれ`best.json`と`metrics.jsonl`に保存しています。step 3,000のlearning rateは5.0000132e-6、step 5,800と6,000は5.0e-6であり、指定した学習率終点が反映されました。

domain validation lossは、generalが4.2887287140、conversationが2.4503843784、medicalが2.3939817746、RealPersonaChatが2.3738891284、MRMPが2.0890000661でした。057のEOS 0.50・3,000 stepと比べると、それぞれ0.004869、0.024385、0.008798、0.018415、0.020348ほど低下しました。また、060のstep 3,000 validation lossは3.3050982118であり、057のEOS 0.50・3,000 stepの3.3050837040とほぼ一致しました。これは、学習率曲線を3,000 stepに固定する変更が、少なくとも3,000 stepまでの比較を壊していないことを示します。

固定48例ではEOS到達が48/48、平均生成長が11.4167 token、token overlap F1が0.233049でした。層別F1はshortが0.323570、mediumが0.203844、longが0.171732でした。057のEOS 0.50・3,000 stepと比べるとoverall F1は0.218087から0.014961上がり、平均生成長は10.1250から1.2917 token伸びました。short F1はわずかに下がりましたが、mediumは0.022503、longは0.027009上がりました。EOS停止率は維持され、未停止はありませんでした。

059の学習率を6,000 stepに延長した条件と比べると、060はoverall F1が0.228467から0.004582上がりました。一方、general・conversation・medical・RPC・MRMPのdomain lossは060の方がそれぞれ0.015625、0.031166、0.010928、0.031511、0.034479高くなりました。つまり、chat-test-v1の重なりF1とdomain validation lossは完全には同じ方向へ動いていません。059の大きなdomain改善は高い学習率を長く維持した影響を含み、060は低い学習率で追加stepを重ねてもchat F1が改善しうることを示しますが、domain保持とのトレードオフが残ります。

固定プロンプト`<|startofconversation|><|speaker:DA|>こんにちはー！<eos:3><|speaker:DC|>`はstep 0、3,000、5,800、6,000のすべてで「こんにちは!」で停止しました。固定promptだけでは条件差を捉えにくいため、全61個の生成文とchat-test-v1の48例を併せて確認します。全stepの生成ファイルは[060の生成サンプル](../../artifacts/samples/issue1-056base-rehearsal-ratio050-eos050-colab-6k-fixedlr3k/)に残しています。

軽量アーカイブのSHA-256は`029609ab5cabd74b4baf5d0685b4e24f1b384b91e53ece52cdb57d407f45e871`、Colab manifestのSHA-256は`f879eae7c54f5f8a9ec8a84d86a2112b172dcafb0d63a8ce06b6c5bd42767bad`、best checkpointのSHA-256は`a331469111129034abbca468a4c33ca0ac4c9e85473edf33ce89627aca9f16da`です。checkpoint本体はGitの追跡対象外ですが、metadata、manifest、生成文、評価JSON/TXTは追跡対象として保存します。

## 次に試すこと

060の結果から、EOS 0.50、rehearsal ratio 0.50、6,000 stepを暫定標準候補としますが、domain lossとchat F1の両方を一つの数値で代表させないことにします。次は、rehearsal ratioを0.25または0.75へ一つずつ変え、一般・医療・会話の保持とchat F1の関係を確認します。その後、条件が固まった段階で20M構造を50Mへ拡大し、reasoning蒸留へ進みます。
