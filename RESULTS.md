# Per-seed results

Validation and test score of every seed; the test score was read once per seed with the best validation checkpoint.

## tgbn-trade (NDCG@10)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.8747168674 | 0.8682526403 | 2,107 | 4.0 | not recorded | Intel Core i5-4690 |
| 1 | 0.8740138345 | 0.8673673756 | 2,107 | 3.9 | not recorded | Intel Core i5-4690 |
| 2 | 0.8745490102 | 0.8683303945 | 2,107 | 3.8 | not recorded | Intel Core i5-4690 |
| 3 | 0.8745484638 | 0.8708152088 | 2,107 | 1.7 | not recorded | AMD Ryzen 9 7950X3D |
| 4 | 0.8750881231 | 0.8686271095 | 2,107 | 2.1 | not recorded | AMD Ryzen 9 7950X3D |

Mean ± SD: validation 0.8745832598 ± 0.0003870939; test 0.8686785458 ± 0.0012835251. Best published test: NAViS 0.863 (read 24 November 2026).

## tgbn-genre (NDCG@10)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.5385130804 | 0.5467318507 | 2,360 | 10.2 | 3.6 | AMD Ryzen 9 7950X3D |
| 1 | 0.5388018262 | 0.5462198179 | 2,360 | 10.3 | 3.6 | AMD Ryzen 9 7950X3D |
| 2 | 0.5386881438 | 0.5466459272 | 2,360 | 10.5 | 3.6 | AMD Ryzen 9 7950X3D |
| 3 | 0.5387219113 | 0.5464159555 | 2,360 | 10.2 | 3.6 | AMD Ryzen 9 7950X3D |
| 4 | 0.5390121722 | 0.5466888131 | 2,360 | 10.0 | 3.5 | AMD Ryzen 9 7950X3D |

Mean ± SD: validation 0.5387474268 ± 0.0001817506; test 0.5465404729 ± 0.0002168381. Best published test: NAViS 0.528 (read 24 November 2026).

## tgbn-reddit (NDCG@10)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.6010157644 | 0.5869283555 | 2,360 | 446.7 | 6.7 | AMD Ryzen 9 7950X3D |
| 1 | 0.6007839141 | 0.5867665327 | 2,360 | 443.0 | 6.5 | AMD Ryzen 9 7950X3D |
| 2 | 0.6008466198 | 0.5867444500 | 2,360 | 451.8 | 6.6 | AMD Ryzen 9 7950X3D |
| 3 | 0.6010304273 | 0.5869831542 | 2,360 | 411.4 | 6.6 | AMD Ryzen 9 7950X3D |
| 4 | 0.6005479108 | 0.5865904951 | 2,360 | 412.4 | 6.6 | AMD Ryzen 9 7950X3D |

Mean ± SD: validation 0.6008449273 ± 0.0001971927; test 0.5868025975 ± 0.0001565927. Best published test: NAViS 0.569 (read 24 November 2026).

## tgbn-token (NDCG@10)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.5021513268 | 0.5156379262 | 2,039 | 420.0 | 11.9 | AMD Ryzen 9 7950X3D |
| 1 | 0.5019063998 | 0.5152554227 | 2,039 | 418.7 | 11.9 | AMD Ryzen 9 7950X3D |
| 2 | 0.5028792434 | 0.5162069020 | 2,039 | 350.4 | 11.9 | AMD Ryzen 9 7950X3D |
| 3 | 0.5021420336 | 0.5162060820 | 2,039 | 349.0 | 11.9 | AMD Ryzen 9 7950X3D |
| 4 | 0.5018506940 | 0.5148644995 | 2,039 | 262.8 | 11.9 | AMD Ryzen 9 7950X3D |

Mean ± SD: validation 0.5021859395 ± 0.0004105874; test 0.5156341665 ± 0.0005896952. Best published test: NAViS 0.513 (read 24 November 2026).

## tgbl-wiki-v2 (MRR)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.8518229017 | 0.8352538397 | 7,995 | 36.2 | not recorded | Intel Core i5-4690 |
| 1 | 0.8518538838 | 0.8350057201 | 7,995 | 35.1 | not recorded | Intel Core i5-4690 |
| 2 | 0.8517093249 | 0.8356125467 | 7,995 | 35.3 | not recorded | Intel Core i5-4690 |
| 3 | 0.8519171625 | 0.8353669766 | 7,995 | 14.7 | not recorded | AMD Ryzen 9 7950X3D |
| 4 | 0.8517102091 | 0.8351666344 | 7,995 | 22.1 | not recorded | AMD Ryzen 9 7950X3D |

Mean ± SD: validation 0.8518026964 ± 0.0000913825; test 0.8352811435 ± 0.0002274375. Best published test: TRACE (per-timestamp) 0.833 ± 0.001 (read 24 November 2026).

## tgbl-review-v2 (MRR)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.4880206495 | 0.5392930140 | 11,642,367 | 34.3 | 3.2 | AMD Ryzen 9 7950X3D |
| 1 | 0.4885889312 | 0.5404769766 | 11,642,367 | 34.3 | 3.2 | AMD Ryzen 9 7950X3D |
| 2 | 0.4877015853 | 0.5387048128 | 11,642,367 | 32.5 | 3.2 | AMD Ryzen 9 7950X3D |
| 3 | 0.4882379665 | 0.5399645684 | 11,642,367 | 32.0 | 3.2 | AMD Ryzen 9 7950X3D |
| 4 | 0.4887223003 | 0.5407502045 | 11,642,367 | 34.9 | 3.2 | AMD Ryzen 9 7950X3D |

Mean ± SD: validation 0.4882542866 ± 0.0004157389; test 0.5398379152 ± 0.0008422515. Best published test: GraphMixer 0.521 ± 0.015 (read 24 November 2026).

## tgbl-flight-v2 (MRR)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.9147641132 | 0.9085960196 | 6,929 | 120.4 | 16.9 | AMD Ryzen 9 7950X3D |
| 1 | 0.9146165152 | 0.9082988670 | 6,929 | 150.2 | 16.9 | AMD Ryzen 9 7950X3D |
| 2 | 0.9148055123 | 0.9086551930 | 6,929 | 149.5 | 16.9 | AMD Ryzen 9 7950X3D |
| 3 | 0.9147491408 | 0.9087258801 | 6,929 | 129.8 | 16.9 | AMD Ryzen 9 7950X3D |
| 4 | 0.9145775703 | 0.9084534849 | 6,929 | 148.4 | 16.9 | AMD Ryzen 9 7950X3D |

Mean ± SD: validation 0.9147025704 ± 0.0000994785; test 0.9085458889 ± 0.0001705987. Best published test: TPNet 0.884 ± 0.001 (read 24 November 2026).

## tgbl-coin-v2 (MRR)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.8537187632 | 0.8693590803 | 12,737 | 205.5 | 9.8 | AMD Ryzen 9 7950X3D |
| 1 | 0.8535767419 | 0.8690435058 | 12,737 | 313.5 | 9.7 | AMD Ryzen 9 7950X3D |
| 2 | 0.8538402380 | 0.8697728836 | 12,737 | 313.1 | 9.8 | AMD Ryzen 9 7950X3D |
| 3 | 0.8533259270 | 0.8689101847 | 12,737 | 310.9 | 9.7 | AMD Ryzen 9 7950X3D |
| 4 | 0.8533582226 | 0.8681707806 | 12,737 | 313.5 | 9.8 | AMD Ryzen 9 7950X3D |

Mean ± SD: validation 0.8535639785 ± 0.0002232961; test 0.8690512870 ± 0.0005938645. Best published test: TPNet 0.832 ± 0.001 (read 24 November 2026).

## thgl-github (MRR)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7558132525 | 0.8472086961 | 7,157 | 62.3 | 6.2 | AMD Ryzen 9 7950X3D |
| 1 | 0.7562429185 | 0.8475072076 | 7,157 | 64.7 | 6.2 | AMD Ryzen 9 7950X3D |
| 2 | 0.7554723371 | 0.8457841778 | 7,157 | 69.8 | 6.2 | AMD Ryzen 9 7950X3D |
| 3 | 0.7558722730 | 0.8460929210 | 7,157 | 75.0 | 6.2 | AMD Ryzen 9 7950X3D |
| 4 | 0.7555388199 | 0.8463888740 | 7,157 | 78.4 | 6.2 | AMD Ryzen 9 7950X3D |

Mean ± SD: validation 0.7557879202 ± 0.0003067671; test 0.8465963753 ± 0.0007349728. Best published test: EdgeBank(unlimited) 0.413 (read 24 November 2026).

## thgl-myket (MRR)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.9164649741 | 0.9145262188 | 7,061 | 251.4 | 17.1 | AMD Ryzen 9 7950X3D |
| 1 | 0.9166610199 | 0.9161149369 | 7,061 | 208.0 | 17.1 | AMD Ryzen 9 7950X3D |
| 2 | 0.9166416145 | 0.9157807009 | 7,061 | 197.5 | 17.1 | AMD Ryzen 9 7950X3D |
| 3 | 0.9165735499 | 0.9161003531 | 7,061 | 189.5 | 17.1 | AMD Ryzen 9 7950X3D |
| 4 | 0.9166145187 | 0.9164409366 | 7,061 | 195.2 | 17.1 | AMD Ryzen 9 7950X3D |

Mean ± SD: validation 0.9165911354 ± 0.0000777893; test 0.9157926293 ± 0.0007454556. Best published test: EdgeBank(unlimited) 0.456 (read 24 November 2026).

## thgl-forum (MRR)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.8170871552 | 0.8164231686 | 8,051 | 820.9 | 12.1 | AMD Ryzen 9 7950X3D |
| 1 | 0.8165564318 | 0.8161186969 | 8,051 | 823.3 | 12.1 | AMD Ryzen 9 7950X3D |
| 2 | 0.8164359159 | 0.8160651043 | 8,051 | 824.6 | 12.1 | AMD Ryzen 9 7950X3D |
| 3 | 0.8168146869 | 0.8163832036 | 8,051 | 776.1 | 12.1 | AMD Ryzen 9 7950X3D |
| 4 | 0.8165075448 | 0.8160686991 | 8,051 | 757.9 | 12.1 | AMD Ryzen 9 7950X3D |

Mean ± SD: validation 0.8166803469 ± 0.0002685620; test 0.8162117745 ± 0.0001765797. Best published test: TGN(edge type) 0.729 ± 0.009 (read 24 November 2026).

## tkgl-polecat (MRR)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.2635056737 | 0.2583569129 | 29,081 | 88.7 | 5.8 | Intel Core Ultra 7 265 |
| 1 | 0.2632220369 | 0.2580690471 | 29,081 | 88.9 | 5.8 | Intel Core Ultra 7 265 |
| 2 | 0.2632003778 | 0.2579334284 | 29,081 | 89.5 | 5.7 | Intel Core Ultra 7 265 |
| 3 | 0.2631730742 | 0.2579571129 | 29,081 | 85.9 | 5.8 | Intel Core Ultra 7 265 |
| 4 | 0.2634061469 | 0.2581276915 | 29,081 | 87.1 | 5.7 | Intel Core Ultra 7 265 |

Mean ± SD: validation 0.2633014619 ± 0.0001463481; test 0.2580888386 ± 0.0001697565. Best published test: CountTRuCoLa 0.256 ± 0.001 (read 24 November 2026).

## tkgl-icews (MRR)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.3471960288 | 0.3454096252 | 34,438 | 523.8 | 27.3 | AMD Ryzen 9 7950X3D |
| 1 | 0.3469795224 | 0.3457213993 | 34,438 | 525.8 | 27.3 | AMD Ryzen 9 7950X3D |
| 2 | 0.3469710448 | 0.3446695779 | 34,438 | 510.2 | 27.2 | AMD Ryzen 9 7950X3D |
| 3 | 0.3468860028 | 0.3449534213 | 34,438 | 512.9 | 27.2 | AMD Ryzen 9 7950X3D |
| 4 | 0.3469862401 | 0.3454161711 | 34,438 | 462.8 | 27.2 | AMD Ryzen 9 7950X3D |

Mean ± SD: validation 0.3470037678 ± 0.0001148899; test 0.3452340390 ± 0.0004179968. Best published test: CountTRuCoLa 0.214 ± 0.001 (read 24 November 2026).

## tkgl-wikidata (MRR)

| seed | validation | test | parameters | wall (min) | peak RSS (GiB) | CPU / GPU |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7405815852 | 0.6703000555 | 81,178 | 18.9 | 22.6 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| 1 | 0.7409226754 | 0.6707893708 | 81,178 | 18.4 | 22.6 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| 2 | 0.7409244276 | 0.6705385718 | 81,178 | 18.3 | 22.6 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| 3 | 0.7408417231 | 0.6708049262 | 81,178 | 18.3 | 22.6 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| 4 | 0.7408622197 | 0.6708697062 | 81,178 | 18.2 | 22.6 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |

Mean ± SD: validation 0.7408265262 ± 0.0001417131; test 0.6706605261 ± 0.0002377081. Best published test: CountTRuCoLa 0.609 ± 0.001 (read 24 November 2026).

