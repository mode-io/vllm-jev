# RadixFlow pressure tests

RadixFlow is an opt-in scheduling feature. These tests compare the complete engineering integration with the ordinary vLLM scheduler using the same Jev plugin and checkpoint. They measure four-candidate decisions, not generated tokens.

## Open-Jev-2B

[Machine-readable results and audit hashes](benchmarks/radixflow-open-jev-2b.json).

Mean ± sample SD over three fresh-service repetitions. Each cell has 32 logical requests per repetition. Short=RACE, long=QuALITY. All four candidates are scored; no generated tokens.

|Input / concurrency|Author req/s|Native vLLM req/s|RadixFlow req/s|RF vs native|RF vs author|RF / native agreement|
|---|---:|---:|---:|---:|---:|---:|
|short / C1|6.45 ± 0.07|7.53 ± 0.16|11.54 ± 0.06|+53.3%|1.79×|100.0%|
|short / C8|6.42 ± 0.11|35.57 ± 0.46|38.91 ± 0.29|+9.4%|6.06×|100.0%|
|short / C32|6.43 ± 0.02|50.19 ± 0.63|50.82 ± 0.77|+1.3%|7.90×|100.0%|
|long / C1|0.68 ± 0.00|4.90 ± 0.06|4.59 ± 0.06|-6.3%|6.73×|100.0%|
|long / C8|0.68 ± 0.00|8.39 ± 0.02|8.77 ± 0.04|+4.5%|12.85×|100.0%|
|long / C32|0.68 ± 0.00|9.36 ± 0.10|9.65 ± 0.02|+3.1%|14.16×|100.0%|

Native vLLM uses the same Jev model/plugin with native scheduling, APC enabled, eager synchronous execution and only read-only measurement hooks. RadixFlow uses the engineering scheduler and complete-state copy guard. Author is the official HTTP server with its default independent-candidate path and built-in serialized predictor lock; its installed runtime falls back to the published PyTorch path because the optional FLA/causal-conv1d kernels are unavailable; no extra serialization was introduced. Native/RF use the same A800 sequentially; author uses a second A800 and may share host CPU resources. Comparisons against author include backend/batching/cache changes; only RF/native isolates the engineering increment.

|Input / C|Method|Mean HTTP ms|p95 HTTP ms|Worker steps|Host ms / step|Core ms / step|Token reuse %|
|---|---|---:|---:|---:|---:|---:|---:|
|short / C1|author|154.73|174.76|—|—|—|0.00|
|short / C1|native|132.50|158.41|80.00|49.85|50.49|45.13|
|short / C1|radixflow|86.40|120.19|48.00|49.50|50.29|45.13|
|short / C8|author|1106.82|1296.39|—|—|—|0.00|
|short / C8|native|214.45|330.11|11.00|76.93|79.17|45.13|
|short / C8|radixflow|199.31|282.68|10.00|74.18|76.77|45.13|
|short / C32|author|2559.15|4736.57|—|—|—|0.00|
|short / C32|native|517.95|602.22|6.00|99.88|103.63|45.13|
|short / C32|radixflow|505.33|575.03|6.00|96.35|100.48|45.13|
|long / C1|author|1464.40|1802.26|—|—|—|0.00|
|long / C1|native|203.75|301.03|82.33|69.03|69.91|84.31|
|long / C1|radixflow|217.57|313.58|82.00|69.97|70.89|84.31|
|long / C8|author|10216.45|13757.97|—|—|—|0.00|
|long / C8|native|902.65|1770.47|35.33|105.26|107.07|84.31|
|long / C8|radixflow|863.30|1661.41|26.00|130.76|133.34|84.31|
|long / C32|author|22963.45|44215.16|—|—|—|0.00|
|long / C32|native|2515.90|3315.55|26.67|124.21|126.41|84.31|
|long / C32|radixflow|2511.55|3243.10|19.33|158.62|162.15|84.31|

## Mechanism notes

Short C1 lowers mean worker calls from 80 to 48 at nearly unchanged host cost (49.85 to 49.50 ms) and 45.13% reuse. No local PENDING recovery is recorded there, so this gain is not attributed to state-ready traversal alone; it includes admission/pooling adaptation in the engineering package. Long C8/C32 record about 20.3/12.7 qualified local recovered rows per run. Long C1 retains almost the same calls and pays added waiting/overhead, with a 6.3% throughput regression. All paired choices agree; probabilities differ (maximum 0.0597 against native, 0.0324 against author).

## Resources and interpretation

Native and RadixFlow each use half of one A800 for the vLLM cache budget; the author uses its original allocations on another full A800. No smaller author memory cap is imposed. Three services per backend are independent repetitions; duplicated questions/visits are not independent system runs. The native/author comparison includes the Jev head plugin/export required to run the same released model, not a different generative checkpoint. Native and RadixFlow use the same eager, synchronous scheduler operating point, not an untouched production default with asynchronous scheduling. GPU1 and GPU2 were released after completion.


## Open-Jev-9B

Mean ± sample SD over three fresh-service repetitions. Each cell has 32 logical requests per repetition. Short=RACE, long=QuALITY. All four candidates are scored; no generated tokens.

|Input / concurrency|Author req/s|Native vLLM req/s|RadixFlow req/s|RF vs native|RF vs author|RF / native agreement|
|---|---:|---:|---:|---:|---:|---:|
|short / C1|4.05 ± 0.05|5.74 ± 0.10|8.12 ± 0.07|+41.4%|2.01×|100.0%|
|short / C8|4.07 ± 0.07|16.22 ± 0.18|16.61 ± 0.13|+2.4%|4.08×|100.0%|
|short / C32|4.03 ± 0.15|17.70 ± 0.17|18.12 ± 0.13|+2.4%|4.49×|100.0%|
|long / C1|0.24 ± 0.00|2.31 ± 0.02|2.22 ± 0.04|-4.0%|9.12×|100.0%|
|long / C8|0.24 ± 0.00|2.98 ± 0.01|2.99 ± 0.02|+0.4%|12.31×|100.0%|
|long / C32|0.24 ± 0.00|3.04 ± 0.02|3.08 ± 0.01|+1.4%|12.69×|100.0%|

Native vLLM uses the same Jev model/plugin with native scheduling, APC enabled, eager synchronous execution and only read-only measurement hooks. RadixFlow uses the engineering scheduler and complete-state copy guard. Author is the official HTTP server with its default independent-candidate path and built-in serialized predictor lock; no extra serialization was introduced. Native/RF use the same A800 sequentially; author uses a second A800 and may share host CPU resources. Comparisons against author include backend/batching/cache changes; only RF/native isolates the engineering increment.

|Input / C|Method|Mean HTTP ms|p95 HTTP ms|Worker steps|Host ms / step|Core ms / step|Token reuse %|
|---|---|---:|---:|---:|---:|---:|---:|
|short / C1|author|246.81|369.60|—|—|—|0.00|
|short / C1|native|173.90|250.23|80.00|66.60|67.23|45.13|
|short / C1|radixflow|122.94|222.34|48.00|73.71|74.55|45.13|
|short / C8|author|1737.02|2084.74|—|—|—|0.00|
|short / C8|native|479.71|864.81|11.00|174.50|176.78|45.13|
|short / C8|radixflow|473.88|828.55|10.00|184.78|187.26|45.13|
|short / C32|author|3973.57|7454.41|—|—|—|0.00|
|short / C32|native|1603.22|1774.50|6.00|294.83|298.79|45.13|
|short / C32|radixflow|1553.89|1690.86|6.00|284.13|288.14|45.13|
|long / C1|author|4114.60|5080.16|—|—|—|0.00|
|long / C1|native|433.13|780.76|82.00|158.73|159.63|84.31|
|long / C1|radixflow|451.13|833.09|82.00|160.45|161.43|84.31|
|long / C8|author|28679.23|38691.73|—|—|—|0.00|
|long / C8|native|2631.39|5688.37|33.00|322.56|324.52|84.31|
|long / C8|radixflow|2621.39|5752.71|24.33|429.65|432.46|84.31|
|long / C32|author|64654.41|124267.53|—|—|—|0.00|
|long / C32|native|7983.01|10423.90|27.00|385.61|388.12|84.31|
|long / C32|radixflow|8160.15|10295.87|19.00|532.99|536.59|84.31|

## Mechanism notes

At short C1, fewer worker calls accompany the throughput gain. Long-input rounds become denser, but their greater host cost absorbs most of the reduction in calls. These are measurements of the whole engineering package, not an isolated traversal effect.

- short C1: mean worker calls 80.00 → 48.00; host ms/call 66.60 → 73.71; qualified local recovered rows 0.00 per run.
- long C8: mean worker calls 33.00 → 24.33; host ms/call 322.56 → 429.65; qualified local recovered rows 17.00 per run.
- long C32: mean worker calls 27.00 → 19.00; host ms/call 385.61 → 532.99; qualified local recovered rows 13.33 per run.

[Machine-readable results and audit hashes](benchmarks/radixflow-open-jev-9b.json). The workload and matched vLLM configuration are identical to the 2B benchmark. All 576 native/RadixFlow paired selected answers agree, as do all 576 author/RadixFlow pairs. The maximum option-probability difference is 0.06946 against native and 0.02490 against author. Model loading is excluded. The author runtime uses its published fallback PyTorch path because optional FLA/causal-conv1d kernels are unavailable.

## OpenJev-0.6B compatibility control

This full-attention model uses its own native branch format and 768-token prefix / 192-token suffix limits. The same 16 short questions are visited twice, with three fresh-service runs per method (576 requests total). Sequence capacity is 16; the round budget is 8192 tokens. No author-runtime speed comparison was made.

| Concurrency | Native req/s | RadixFlow req/s | Gain vs native | Choice agreement |
|---:|---:|---:|---:|---:|
| 1 | 22.59 ± 0.22 | 31.21 ± 0.42 | +38.2% | 93.75% |
| 8 | 150.96 ± 5.52 | 132.47 ± 19.99 | -12.2% | 94.79% |
| 32 | 153.23 ± 3.34 | 142.74 ± 4.69 | -6.9% | 95.83% |

| C | Method | Mean HTTP ms | Worker steps | Host ms / step | Token reuse % |
|---:|---|---:|---:|---:|---:|
| 1 | native | 44.07 | 63.00 | 19.13 | 82.49 |
| 1 | radixflow | 31.87 | 32.00 | 20.23 | 82.49 |
| 8 | native | 49.60 | 9.00 | 20.56 | 82.49 |
| 8 | radixflow | 56.23 | 10.00 | 20.34 | 82.49 |
| 32 | native | 133.82 | 9.00 | 20.23 | 82.49 |
| 32 | radixflow | 136.69 | 9.00 | 20.36 | 82.49 |

The matched outputs differ on 15/288 paired responses. Native top-two probability margins for these responses range from 0.0037 to 0.0198; the largest option-probability difference is 0.04313. This identifies near-tie cases but does not establish their cause or semantic equivalence. This model is a compatibility control, not a recommended RadixFlow deployment result. [Summary and audit hashes](benchmarks/radixflow-openjev-06b.json).

## Reproduction

Use [RadixFlow setup](radixflow.md) to build the Rust index and enable the scheduler. Freeze identical checkpoint, dtype, prompt tokens, cache memory, sequence budget, token budget and workload across native and RadixFlow runs. Alternate their execution order across fresh-service repetitions. Warm-up uses separate cache namespaces and is excluded.

The matched vLLM configuration is vLLM 0.29.0, BF16 base weights, FP32 score head, eager synchronous execution, native prefix caching, 64 sequence slots and an 8192-token round/context limit. Input selection uses calibration data, no gold labels and no truncation. Actual branch IDs, scheduled token intervals, cache-token usage, complete worker calls and GPU ownership were audited. Repeated scheduled intervals are retained rather than deduplicated. Raw models, input payloads and server logs stay outside the repository.
