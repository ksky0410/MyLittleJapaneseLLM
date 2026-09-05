# 実験049：FineWeb2を共通generalにしたIssue #1 source ablation

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。Issue #1では、標準文、RealPersonaChat、MRMPの組み合わせを比較し、会話を通常pretrainingへ混ぜる場合とSFTへ使う場合を分けることが提案されています。実験048ではその比較のために青空文庫をgeneral sourceへ使いましたが、青空文庫が先に枯渇し、条件ごとにmedical比率が変わることが分かりました。049ではFineWeb2 Edu Japaneseを共通general sourceへ差し替え、source差と医療比率の差を分けます。

比較条件はcore、rpc、mrmp、bothの4つです。coreはFineWeb2 Edu Japanese + 医療、rpcはFineWeb2 Edu Japanese + 医療 + RealPersonaChat、mrmpはFineWeb2 Edu Japanese + 医療 + MRMP、bothはFineWeb2 Edu Japanese + 医療 + RealPersonaChat + MRMPです。希望weightはそれぞれcoreが9:1、rpc/mrmpが8:1:1、bothが8:1:0.5:0.5です。全条件でtarget 1,000,000 Token、seed 42、Tokenizer、モデル、学習stepを固定します。論理単位を途中分割・複製しないため、実測比率はmanifestの値を優先します。

仮説は、RealPersonaChat単独では長めの1対1雑談や話題継続が、MRMP単独では短い相づち、mention、複数話者markerが相対的に学習されるというものです。bothは両方の利点を持つ可能性がありますが、会話sourceを増やすことでgeneralまたはmedical validationが悪化する可能性もあります。loss、Token overlap、EOSだけでは意味的な自然さを保証しないため、生成TXTを全条件・全stepで保存します。

## データとモデル

general sourceは`artifacts/corpus/fineweb2-edu-japanese-v1/train.txt`、入力manifestに記録された本文SHA-256は`471869caa73aa5987a52a2dbcfa28846441d0729ff03bb0c05db0fa461e3890f`です。medical sourceは`artifacts/corpus/medical-qb-v2/train.txt`、会話sourceは実験048で分離した`artifacts/corpus/conversation-sources-v1/real-persona-chat-train.txt`と`mrmp-train.txt`です。医師国家試験の原本`/Users/koseki/projects/medilink_analysis`は読み取り専用で扱い、変更しません。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。モデルは約5M parameter、dim 240、6層、6 heads、context length 256、absolute position embedding、LayerNorm、GELUです。学習はbatch size 8、500 step、AdamW、learning rate 3e-4から3e-5、warmup 50、weight decay 0.1、seed 42、eval/sample interval 100で実施します。CPUで実施した場合は速度をGPUと比較しません。

各条件のconfigは`configs/issue1-core-5m-smoke.toml`、`issue1-rpc-5m-smoke.toml`、`issue1-mrmp-5m-smoke.toml`、`issue1-both-5m-smoke.toml`です。049を開始する基準commitは`e89b6ed`です。configのSHA-256はcore `c809448ffc69c9635c9bef6e0ff779dadaf12d28a089559982aacb8686625534`、rpc `9035314c385b42a47b76a6fb672ba14165c51fa8a03d499f03a9e65c2f47f061`、mrmp `97cbc4d907ad8d2c2cd87c36474e90e75a790fbd98545ff24f6cbce56b2e18bf`、both `f8cfd4e72c2fc00b9d7b3c5c69d4ca3ced0916bdacafca04b62a9e8bbbbbba73`です。FineWeb2 train manifestのSHA-256は`e29cd2cf303c6b0835d46923a0e88ab8dfcabbab7d760c1e404e8027f5e2d9fb`、medical manifestは`b9f6c0f4723ca18341f5fac857aa21244d508cbe81779c249aca1318cdb01142`、source分離manifestは`aa4aba02c566a79aa49396252f39a365fa4aa912165410a73a2b89df0d5865f1`です。派生train textのSHA-256はRealPersonaChat `09e6db9fb871bdad8315293a812e06d9d64d1ddacfd4643eac98a9a8f2e7ebee`、MRMP `9220f556356d68c60bb1af3d97c756e8d9c7daf3fde5623faad155c36bdd8c0c`です。validationはgeneral、medical、conversation全体に加え、RealPersonaChatとMRMPをsource別にToken化した列を使います。固定chat-testは`experiments/evaluation/chat-test-v1.json`の48例です。

## 実行コマンド

```bash
.venv/bin/python scripts/split_conversation_sources.py \
  --input-dir artifacts/corpus/conversation-v1 \
  --output-dir artifacts/corpus/conversation-sources-v1

.venv/bin/python scripts/mix_corpora.py \
  --source general=artifacts/corpus/fineweb2-edu-japanese-v1/train.txt \
  --source medical=artifacts/corpus/medical-qb-v2/train.txt \
  --weight general=9 --weight medical=1 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --target-tokens 1000000 --seed 42 \
  --output artifacts/corpus/issue1-core-1m-fineweb.txt \
  --manifest artifacts/corpus/issue1-core-1m-fineweb.manifest.json

.venv/bin/python scripts/mix_corpora.py \
  --source general=artifacts/corpus/fineweb2-edu-japanese-v1/train.txt \
  --source medical=artifacts/corpus/medical-qb-v2/train.txt \
  --source rpc=artifacts/corpus/conversation-sources-v1/real-persona-chat-train.txt \
  --weight general=8 --weight medical=1 --weight rpc=1 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --target-tokens 1000000 --seed 42 \
  --output artifacts/corpus/issue1-rpc-1m-fineweb.txt \
  --manifest artifacts/corpus/issue1-rpc-1m-fineweb.manifest.json

.venv/bin/python scripts/mix_corpora.py \
  --source general=artifacts/corpus/fineweb2-edu-japanese-v1/train.txt \
  --source medical=artifacts/corpus/medical-qb-v2/train.txt \
  --source mrmp=artifacts/corpus/conversation-sources-v1/mrmp-train.txt \
  --weight general=8 --weight medical=1 --weight mrmp=1 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --target-tokens 1000000 --seed 42 \
  --output artifacts/corpus/issue1-mrmp-1m-fineweb.txt \
  --manifest artifacts/corpus/issue1-mrmp-1m-fineweb.manifest.json

.venv/bin/python scripts/mix_corpora.py \
  --source general=artifacts/corpus/fineweb2-edu-japanese-v1/train.txt \
  --source medical=artifacts/corpus/medical-qb-v2/train.txt \
  --source rpc=artifacts/corpus/conversation-sources-v1/real-persona-chat-train.txt \
  --source mrmp=artifacts/corpus/conversation-sources-v1/mrmp-train.txt \
  --weight general=8 --weight medical=1 --weight rpc=0.5 --weight mrmp=0.5 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --target-tokens 1000000 --seed 42 \
  --output artifacts/corpus/issue1-both-1m-fineweb.txt \
  --manifest artifacts/corpus/issue1-both-1m-fineweb.manifest.json

.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model --input artifacts/corpus/issue1-core-1m-fineweb.txt --output artifacts/tokens/issue1-core-1m-fineweb-train.bin
.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model --input artifacts/corpus/issue1-rpc-1m-fineweb.txt --output artifacts/tokens/issue1-rpc-1m-fineweb-train.bin
.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model --input artifacts/corpus/issue1-mrmp-1m-fineweb.txt --output artifacts/tokens/issue1-mrmp-1m-fineweb-train.bin
.venv/bin/python scripts/encode_data.py --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model --input artifacts/corpus/issue1-both-1m-fineweb.txt --output artifacts/tokens/issue1-both-1m-fineweb-train.bin
.venv/bin/python scripts/train_torch.py \
  --config configs/issue1-core-5m-smoke.toml \
  --device cpu
```

残り3条件もconfigだけを置き換えて実行します。学習開始前に、入力source、target token数、Token列hash、config hash、基準Git commitを記録します。学習後は最終および最良checkpointをreloadし、全domain評価と固定chat-testを実行します。

13:51 JST、4条件の混合を実行しました。FineWeb2をgeneral sourceにしたことでsource枯渇は起きず、selected token countはcore 999,987、rpc 999,974、mrmp 999,978、both 999,970でした。実測Token比率はcoreがgeneral 90.0095%・medical 9.9905%、rpcがgeneral 80.0237%・medical 9.9907%・rpc 9.9857%、mrmpがgeneral 80.1349%・medical 9.9906%・mrmp 9.8745%、bothがgeneral 80.1587%・medical 10.0079%・rpc 4.9710%・mrmp 4.8623%でした。048で見つかったgeneral枯渇とmedical比率の交絡は解消できています。

混合本文のSHA-256はcore `5a817e8d25af5f7e9d8e5ffbd27e1d482f50f3c544eed0e9fa4e5520418554f4`、rpc `a9b29e4f4ba813e35740f1cb3cdc8fd69ed0727048007b7b069c45325015e7c8`、mrmp `0add72482bcba2a8ec5d0319ec3812f1de47ccfdb60e10a2dd548472c92ea12f`、both `76d11a5259c83a6863f42bc330b7d0b3e93d63e9f310c38d3ad500663c19d71e`です。混合manifestは入力・Tokenizer・source別採用数とToken比率を保持し、学習前のデータ条件として保存します。続けて4条件をToken化します。

13:53 JST、4条件のtrain Token列とsource別validation Token列を作成しました。train Token数とSHA-256はcore 999,987 / `ebca09587890bfbfb76b6a0d968b198be55943993fc011115a1736d88914e9a4`、rpc 999,974 / `24b61c79c6144e74e8e0598d54182c7a8f109ab75fda0e7d79eea90d68c268b7`、mrmp 999,978 / `9dacab60e0b483526f9f61f2f4254290c86c48a24a4c13abcb2922b38a1c100c`、both 999,970 / `758b46f6bb946afd7e2c3604714db71166d79564f8c652e8cc950b23d3338879`です。source別validationはRealPersonaChat 948,172 Token / `30f8b66828b7a5c1171024af220613cc79d066089a4c09896456112f8754491c`、MRMP 156,475 Token / `9431d4d7432f89f69d9656bf2c3eea7c18dbcb94e84467060cba3fa8d9445623`です。これらは学習・評価条件として記録し、元のJSONLは変更していません。

次の実行は各configを同じseedで500 step学習するCPU探索です。予定コマンドは`.venv/bin/python scripts/train_torch.py --config configs/issue1-core-5m-smoke.toml --device cpu`、rpc・mrmp・bothも同じ形式です。4条件の学習前記録とToken hashを確定したため、ここから学習を開始します。

## 成功条件

4条件の混合とToken化が入力hashの検証つきで完了し、各500 step学習がNaN、OOM、shape error、Token列不足なしに完走することです。各条件について、metrics、checkpoint metadata、step 0・100・200・300・400・500の生成文、general・conversation・medical・RPC・MRMP評価、固定chat-test JSON/TXTを保存します。性能差が小さい場合も失敗とはせず、会話sourceの影響がこの規模では検出できなかった結果として記録します。

## 実験中の記録

データ加工、Token化、条件ごとの開始・途中・終了、エラー、生成文、評価結果をこの節へ作業中に追記します。生成文は自然さに関係なく削除しません。

## 結果と解釈

core・rpc・mrmp・bothの各条件を、実測source別Token比率、一般・source別validation loss、固定chatのEOS・生成長・Token overlap、人手レビュー用の生成本文に分けて解釈します。医療validationが低くても医学的正確性を意味しないため、医師国家試験データ由来の見かけの専門性と一般会話能力を混同しません。

## 次に試すこと

049でsource差が見えた条件について、同じ条件の学習stepまたはToken予算を増やします。差が見えない場合は、通常pretrainingへの少量混合より、会話履歴と応答本文を分けたSFT、短文sampling、rehearsalの比較を優先します。その後、Issue #1の固定promptとheld-out生成について文脈適合・役割適合・崩壊を人手レビューし、自動指標を補います。
