# Issue #1：現代的な会話日本語を今後の実験へつなぐ記録

## Issueの確認

2026-09-05に、GitHub Issue [#1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)の本文を確認しました。Issueの目的は、青空文庫のような整った文章だけでなく、現代の自然なチャット・雑談日本語を学習へ加えることです。候補として、1対1の雑談を含むRealPersonaChatと、関係性つきの複数人チャットを含むMulti-Relational Multi-Party Chat Corpus（MRMP）が挙げられています。

Issueが特に重視している観察対象は、助詞、語尾、相槌、短文、割り込み、mention、砕けた表現、スラング、会話ターンの自然さです。また、標準文だけ、標準文とRealPersonaChat、標準文とRealPersonaChat・MRMPの両方を比較し、事前学習へ混ぜる場合と会話形式SFTとして使う場合を分けることが提案されています。データのライセンスと出所を記録し、元データを大量にGitへ追加せず、話者境界を保存し、固定promptと生成例を残すこともIssueの条件です。

Issue末尾には「まじで」「それな」「今日なにしてた？」「やば」「なんかさ」「いやそれは」「おつかれ」「明日ひま？」という固定promptが挙げられています。ただし、元データは2022年前後の収集が中心であるため、2026年時点の最新スラングを代表する資料とはみなしません。実在の個人LINE履歴を追加する場合も、本人の明確な同意、匿名化、利用範囲の確認が別途必要であり、公開データの実験と混ぜません。

## Issueに対して実装済みのこと

RealPersonaChatとMRMPは、公開元のcommit、利用条件、source名をmanifestへ残したうえで、会話単位のtrain・validation・testへ分割しました。元リポジトリはsmall_llm側へコピーせず、会話IDと話者境界を保存した加工結果だけを使っています。医師国家試験データも同じ一般日本語の混合コーパスへ補助的に加え、医療専用モデルへはしていません。元のSQLiteは読み取り専用で扱い、`/Users/koseki/projects/medilink_analysis`側を変更していません。

通常のnext-token pretrainingへ会話を混ぜる経路と、履歴・話者markerを入力にし、応答本文とEOSだけへlossをかけるSFT経路を分けました。SFTでは一般・会話・医療の忘却を確認するためrehearsalも実装し、短文samplingの効果も独立条件で比較しました。Issueの固定promptは`experiments/prompts/issue-1-chat-v1.json`と`issue-1-chat-sft-v1.json`へ保存しています。

評価については、validationを学習中の監視へ再利用しないようtestから一会話一例で48例を固定しました。応答長をshort・medium・longの各16例へ分け、各層でMRMPとRealPersonaChatを8例ずつ含め、manifestへsource、話者、履歴長、context切り詰め、train本文完全一致候補を保存しています。Token overlap、EOS、生成長は自動評価しますが、意味や役割の適合は別物であるため、`scripts/create_chat_review_template.py`で人手判定欄を未記入のまま生成できるようにしました。

## 今後のIssue #1候補

第一候補は、今回の固定48例へ人手レビューを入力し、文脈適合、応答役割、明らかな崩壊を条件ごとに比較することです。自動指標が改善しても、短くなっただけなのか、話題へ適切に応答できたのかを分けて判断します。train本文との完全一致候補7例と履歴切り詰め33例は、レビュー時に影響要因として扱います。

第二候補は、標準文のみ、標準文+RealPersonaChat、標準文+RealPersonaChat+MRMPを、同じTokenizer、同じ総Token予算、同じモデル、同じstep、同じseedで比較するsource ablationです。現行の会話v1は両sourceを含むため、次はRealPersonaChat単独sourceを分けた混合manifestを作り、source追加の効果と会話量の効果を分離します。医療sourceはこの比較では固定し、会話sourceだけを差分にします。

第三候補は、会話を通常pretrainingへ混ぜる条件と、会話SFTへ使う条件を同じ固定testで比較することです。会話全体へ一様にlossをかけたときに話者markerや文体だけを学習してしまうのか、応答側maskが内容対応へ寄与するのかを確認します。rehearsal ratioと短文samplingは同時に変えず、別の実験番号で管理します。

第四候補は、短文・相槌・質問・同意/不同意・誘い・別れなどの応答機能を、target本文の長さだけではなく簡易カテゴリでも層別化することです。カテゴリは機械的な仮ラベルとして記録し、意味の判定と混同しないようにします。固定promptの8例だけでなく、会話testから複数例を抽出し、カテゴリごとの失敗を確認します。

## 現在の優先順位

まず実験029でFineWeb混合コーパスの学習step不足を検証し、その後、同じデータ条件を固定して実験030の20M級モデルへ進みます。容量を増やしたあとに、Issue #1の標準文・RealPersonaChat・MRMP source ablationと人手レビューを行います。RoPE、RMSNorm、SwiGLUなどの現代的な構成は、20M baselineとIssue #1の評価が揃ってから一つずつ導入します。

どの候補でも、元データを削除・上書きせず、取得元、license、commitまたは取得日時、入力hash、加工方法、Tokenizer hash、Token数、学習条件、生成文を実験ノートとGitHubの軽量artifactへ残します。失敗した条件や崩れた生成も削除せず、後から比較できる状態を保ちます。
