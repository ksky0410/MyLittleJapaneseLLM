# 実験084：Issue #1固定プロンプトによる現行モデルの会話評価

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。GitHub Issue #1で提案されている現代的な会話日本語を、既存の48例chat-testとは別の固定プロンプトで評価します。Issue #1のプロンプトは「まじで」「それな」「今日なにしてた？」「やば」「なんかさ」「いやそれは」「おつかれ」「明日ひま？」です。これまでの評価ではIssue #1由来のRPC・MRMPデータを含むSFTの領域lossとchat-testを比較してきましたが、Issueに書かれた口語表現そのものへの応答をまだ直接測っていませんでした。

今回の主対象は、081の反復事前学習checkpointからIssue #1のRPC・MRMPを合わせてSFTした082です。比較対象として、同じデータからSFTした078と、EOS loss weightを0.0にした083も評価します。各モデルについて、同じtokenizer、同じプロンプト、同じseed、同じ生成長・温度・top-kで`raw`形式と`conversation`形式の両方を保存します。

仮説は、082がIssue #1の口語固定プロンプトに対して、078より応答の形式と終了を安定させ、083より自然で短くまとまった日本語を生成することです。成功条件は、082の固定プロンプト評価で、48例chat-testの結果だけでは見えない口語表現への適合を確認でき、083のような冗長化・反復・終了不能が増えていないことです。これは新しい学習ではなく評価実験ですので、結果に基づいて次のデータ量・反復学習・SFT条件を決めます。強いモデルからの蒸留は今回も使いません。

## 再現条件

評価対象は次の3 checkpointです。

* 078：`artifacts/checkpoints/issue1-both-50m-sft-from-5m-seed123-3k/best.pt`
* 082：`artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-3k/best.pt`
* 083：`artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k/best.pt`

使用するIssue #1固定プロンプトは`experiments/prompts/issue-1-chat-v1.json`で、SHA-256は`b538af4f00668e60e712aa796b1de5d51e0f677c8b9a19bc4445d40a90929594`です。082の設定ファイルは`configs/issue1-both-50m-sft-from-5m-two-pass-seed123-3k.toml`、SHA-256は`c64652986d1b2b9efd75674902e59c2640e43ca8604ebaa3c42d2feffaed61d5`です。078の設定ファイルは`configs/issue1-both-50m-sft-from-5m-seed123-3k.toml`、SHA-256は`7932e93671c901caac1fcdadfa58f80c48faa3d158f0e22a5d5109c9dbfadba3`です。083の設定ファイルは`configs/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k.toml`、SHA-256は`cd2b7f0df1a01073f764556891b8707ba996f6b7272bf0db5efae5310e402a2f`です。

評価設定は、最大生成64 token、temperature 0.8、top-k 40、seed 42、CPU推論です。`raw`ではプロンプトをそのまま入力し、`conversation`では評価スクリプトの会話テンプレートを使います。JSONには各生成結果と集計値、TXTには人間が確認するための全文を保存します。

実行コマンドは次の6本です。

```bash
uv run python scripts/evaluate_torch_prompt_set.py --config configs/issue1-both-50m-sft-from-5m-seed123-3k.toml --checkpoint artifacts/checkpoints/issue1-both-50m-sft-from-5m-seed123-3k/best.pt --prompt-file experiments/prompts/issue-1-chat-v1.json --template raw --max-new-tokens 64 --temperature 0.8 --top-k 40 --seed 42 --device cpu --output artifacts/evaluations/issue1-both-50m-sft-from-5m-seed123-3k-issue1-prompts-raw.json --text-output artifacts/evaluations/issue1-both-50m-sft-from-5m-seed123-3k-issue1-prompts-raw.txt
uv run python scripts/evaluate_torch_prompt_set.py --config configs/issue1-both-50m-sft-from-5m-seed123-3k.toml --checkpoint artifacts/checkpoints/issue1-both-50m-sft-from-5m-seed123-3k/best.pt --prompt-file experiments/prompts/issue-1-chat-v1.json --template conversation --max-new-tokens 64 --temperature 0.8 --top-k 40 --seed 42 --device cpu --output artifacts/evaluations/issue1-both-50m-sft-from-5m-seed123-3k-issue1-prompts-conversation.json --text-output artifacts/evaluations/issue1-both-50m-sft-from-5m-seed123-3k-issue1-prompts-conversation.txt
uv run python scripts/evaluate_torch_prompt_set.py --config configs/issue1-both-50m-sft-from-5m-two-pass-seed123-3k.toml --checkpoint artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-3k/best.pt --prompt-file experiments/prompts/issue-1-chat-v1.json --template raw --max-new-tokens 64 --temperature 0.8 --top-k 40 --seed 42 --device cpu --output artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-3k-issue1-prompts-raw.json --text-output artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-3k-issue1-prompts-raw.txt
uv run python scripts/evaluate_torch_prompt_set.py --config configs/issue1-both-50m-sft-from-5m-two-pass-seed123-3k.toml --checkpoint artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-3k/best.pt --prompt-file experiments/prompts/issue-1-chat-v1.json --template conversation --max-new-tokens 64 --temperature 0.8 --top-k 40 --seed 42 --device cpu --output artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-3k-issue1-prompts-conversation.json --text-output artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-3k-issue1-prompts-conversation.txt
uv run python scripts/evaluate_torch_prompt_set.py --config configs/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k.toml --checkpoint artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k/best.pt --prompt-file experiments/prompts/issue-1-chat-v1.json --template raw --max-new-tokens 64 --temperature 0.8 --top-k 40 --seed 42 --device cpu --output artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k-issue1-prompts-raw.json --text-output artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k-issue1-prompts-raw.txt
uv run python scripts/evaluate_torch_prompt_set.py --config configs/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k.toml --checkpoint artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k/best.pt --prompt-file experiments/prompts/issue-1-chat-v1.json --template conversation --max-new-tokens 64 --temperature 0.8 --top-k 40 --seed 42 --device cpu --output artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k-issue1-prompts-conversation.json --text-output artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k-issue1-prompts-conversation.txt
```

## 実験中の記録

開始前のGit commitは`7383035`です。作業ツリーに未コミット変更がないことを確認してから評価を開始します。評価中にエラー、checkpoint読込不良、空出力、EOS未到達、文字崩れがあれば、そのままノートへ追記します。生成文は省略せず、評価TXTとJSONをGit管理下に置きます。

## 実験終了後の結果と解釈

評価は2026年9月6日にローカルCPUで完了しました。6条件ともcheckpointを正常に読み込み、8プロンプトすべての生成文をJSONとTXTへ保存しました。評価中のエラー、空の生成、ファイル欠落はありませんでした。

082のraw形式はEOS 6/8、平均生成長23.125 tokenでした。「まじで」への「、「あひたいな!」と聞いた。」や、「いやそれは」から古典的な物語文へ流れる出力があり、Issue #1の口語応答としては不十分でした。conversation形式ではEOS 8/8、平均生成長4.000 tokenでしたが、8例中7例が「こんにちは」「こんばんは」「よろしくお願いします」系で、入力の意味に応答せず、会話テンプレートを学習しただけの状態です。「明日ひま？」に対して「こんにちはー。」と返すため、終了制御は安定しても自然な対話能力はまだ得られていません。

078はraw形式でEOS 7/8、平均生成長14.125 token、conversation形式でEOS 8/8、平均生成長4.125 tokenでした。082よりrawの長さは短いものの、conversation形式の応答は082と同様に挨拶へ偏りました。081の反復事前学習でvalidation lossが改善しても、Issue #1の口語固定プロンプトに対応する知識と応答の多様性が十分に増えたとはいえません。

083はraw形式でEOS 3/8、平均生成長47.000 token、conversation形式でEOS 7/8、平均生成長17.625 tokenでした。rawでは長い出力、古風な語彙、文脈からの逸脱が増え、conversation形式でも「こんにちは!よろしくお願いします!」の反復と、長く止まらない生成が現れました。083のEOS loss weight 0.0は、長く生成させるだけで自然さを改善しないという084の固定プロンプトでも再確認されました。

この結果から、082を現在の標準SFT条件として保持します。ただし、082を「自然な会話ができるモデル」とは判定しません。モデルは会話形式の終了記号と定型挨拶を覚えていますが、Issue #1の口語入力に対する意味的な応答、相づち、誘いへの返答、反対意見への応答が不足しています。今後はEOS重みの微調整だけでなく、会話データを一般日本語の中へどう混ぜるか、会話応答の多様性をどう増やすか、SFT前に一般データを何トークン反復するかを主な研究対象にします。

固定プロンプト評価の全文は、モデル別・template別に次のTXTへ保存しています。

* 078 raw：`artifacts/evaluations/issue1-both-50m-sft-from-5m-seed123-3k-issue1-prompts-raw.txt`
* 078 conversation：`artifacts/evaluations/issue1-both-50m-sft-from-5m-seed123-3k-issue1-prompts-conversation.txt`
* 082 raw：`artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-3k-issue1-prompts-raw.txt`
* 082 conversation：`artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-3k-issue1-prompts-conversation.txt`
* 083 raw：`artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k-issue1-prompts-raw.txt`
* 083 conversation：`artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k-issue1-prompts-conversation.txt`

JSONとTXTのSHA-256は次のとおりです。

* 078 raw：JSON `5dd239926b012fad95120d43f09eeabe539a7662ad8654bdb432463098dd4c8f`、TXT `f8c5bf7ca19cd9139e13ed5f36ec006b521c79ea6c3e4e0307f938a46bbb4e1e`
* 078 conversation：JSON `621d7c110c6c82f0a81045c2d5d0d009fbc9ccbfc58e575d81d906bac53c57dd`、TXT `f5f0e03542d2d97834aa8bf29a01de7b43289a41b7b5bd8826d5315150cd23fd`
* 082 raw：JSON `a3b3b10f9bb8aeafb1eec2bd1e219d345523a92e4e2e08d4f4166f7cba2d5e0b`、TXT `36a453dc33afd126db98602304de39a83c60fb8fcb004baaaee8b619ffc92fb8`
* 082 conversation：JSON `b8c13d52d1cd17a6f02dc6b8e669bc5f2c013b21d0eddd1487f0329507ae8be1`、TXT `0427c7776f5762e601a4081a13c83bbb4e8b02480daa313b2d62ebac3f4a3b41`
* 083 raw：JSON `976d5361cfef668a869fee15872264d7404ef1f05c1a0a9e8461025d927424a8`、TXT `e393a1b84f2819339293857647dcaad41474bf0c87c711d3c3ef37e97bc3d2bf`
* 083 conversation：JSON `9b5cfcad44a9ec705016eab19629e017904b64e88204ec27c53e264e0bb6612d`、TXT `ce5f2564de2ec4e2a340140937007618609ff8401e9bd9d63a513c1be317da6e`

## 次に試すこと

Issue #1固定プロンプトで082の弱点が明確になったため、次は蒸留を使わず、一般日本語を含む事前学習量を増やした50M基盤へ、RPCとMRMPの会話SFTを再度行う候補を第一にします。具体的には、081の5M Token反復をさらに増やした基盤、または既存の10M Token列を複数周回した基盤を用意し、082と同じSFT条件で比較します。これにより「会話データを増やす前に一般日本語の流暢さが足りない」のか、「会話SFTの応答形式が単調」なのかを分けます。

第二候補は、RPC単独・MRMP単独ではなく、両方を同じ応答token予算で使いながら、短い定型挨拶の比率を下げ、応答内容の多様性を保つSFTデータを作ることです。EOS weight 0.0は不採用とし、まず082の0.5を固定してデータ側の改善を検証します。実験085ではIssue #1固定プロンプトを開始前・終了後評価へ組み込み、学習中の生成も同じ8例で保存します。
