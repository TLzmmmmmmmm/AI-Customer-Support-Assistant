# Week 4 Day 2 Metrics, Cost, and Performance Analysis

## Workload methodology

Sequential production-style HTTP workload with no runner retries. Manifest seed: `20260914`; repeat count: `5`. Target routes shape the workload only; every grouping below uses the actual production route.

## Sample counts

| Attempted | Successful | Failed | Skipped | Single-turn success | Multi-turn success |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 132 | 132 | 0 | 0 | 120 | 12 |

## Metric definitions

Success requires a uniquely joined production summary with HTTP 200 and `outcome=success`. P50 and P95 use the nearest-rank definition; each table includes its contributing sample count. Stage measurements may overlap and are not added as percentages of total latency.

## Telemetry coverage

| Matched | Missing request ID | Missing summary | Duplicate summary | Unrelated summaries |
| ---: | ---: | ---: | ---: | ---: |
| 132 | 0 | 0 | 0 | 6 |

## P50/P95 latency

| Sample count | Mean ms | P50 ms | P95 ms |
| ---: | ---: | ---: | ---: |
| 132 | 1317.9 | 1000.0 | 3156.0 |

## Latency by actual route

| Route | Sample count | Mean ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: |
| contact | 21 | 919.0 | 844.0 | 1437.0 |
| direct | 20 | 993.6 | 906.0 | 1359.0 |
| exact_product | 22 | 1073.0 | 937.0 | 1375.0 |
| fallback | 24 | 529.2 | 15.0 | 2031.0 |
| knowledge | 21 | 1892.2 | 1781.0 | 2532.0 |
| product_search | 24 | 2448.0 | 2235.0 | 3688.0 |

## Stage diagnosis

`dominant_stage_ratio` is the largest non-null measured stage divided by positive total latency. It identifies whether one measured stage dominates; stage timings may be sequential or overlap.

| Route | Stage | Sample count | Mean ms | P50 ms | P95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| contact | router | 21 | 0.0 | 0.0 | 0.0 |
| contact | retrieval | 0 | — | — | — |
| contact | tool | 21 | 0.0 | 0.0 | 0.0 |
| contact | model | 21 | 911.5 | 844.0 | 1437.0 |
| direct | router | 20 | 0.0 | 0.0 | 0.0 |
| direct | retrieval | 0 | — | — | — |
| direct | tool | 0 | — | — | — |
| direct | model | 20 | 982.8 | 891.0 | 1359.0 |
| exact_product | router | 22 | 0.0 | 0.0 | 0.0 |
| exact_product | retrieval | 0 | — | — | — |
| exact_product | tool | 22 | 0.0 | 0.0 | 0.0 |
| exact_product | model | 22 | 1058.3 | 922.0 | 1359.0 |
| fallback | router | 24 | 522.2 | 0.0 | 2015.0 |
| fallback | retrieval | 0 | — | — | — |
| fallback | tool | 0 | — | — | — |
| fallback | model | 8 | 1566.5 | 922.0 | 3953.0 |
| knowledge | router | 21 | 0.0 | 0.0 | 0.0 |
| knowledge | retrieval | 21 | 215.9 | 187.0 | 406.0 |
| knowledge | tool | 0 | — | — | — |
| knowledge | model | 21 | 1665.9 | 1625.0 | 2360.0 |
| product_search | router | 24 | 32.6 | 0.0 | 0.0 |
| product_search | retrieval | 23 | 233.0 | 157.0 | 438.0 |
| product_search | tool | 23 | 233.0 | 157.0 | 438.0 |
| product_search | model | 24 | 2207.8 | 1984.0 | 3454.0 |

## Tokens/request

Complete token sample count: 132 of 132; excluded: 0.

| Average input | Average output | Average total |
| ---: | ---: | ---: |
| 1994.2 | 110.4 | 2104.6 |

## Tokens by route

| Route | Sample count | Average input | Average output | Average total |
| --- | ---: | ---: | ---: | ---: |
| contact | 21 | 1825.2 | 22.0 | 1847.2 |
| direct | 20 | 1669.0 | 21.9 | 1690.8 |
| exact_product | 22 | 2162.1 | 44.8 | 2206.9 |
| fallback | 24 | 90.5 | 34.9 | 125.5 |
| knowledge | 21 | 3138.3 | 170.9 | 3309.1 |
| product_search | 24 | 3161.7 | 344.5 | 3506.2 |

## Tool execution

Average execution count: 0.5.

Execution-count distribution: {"0": 66, "1": 66}

Ordered executed-tool sequences: [{"tool_names": ["search_products"], "count": 23}, {"tool_names": ["get_product_details"], "count": 22}, {"tool_names": ["get_contact_info"], "count": 21}]

Product-search requests with more than one execution: 0.

## Estimated cost/request

Coverage: 132 of 132 successful requests; excluded: 0. Currency: CNY.

Pricing snapshot for `deepseek-v4-flash` per 1,000,000 tokens:

| Period | Cache-hit input | Cache-miss input | Output |
| --- | ---: | ---: | ---: |
| Off-peak | 0.02 | 1.00 | 4.00 |
| Peak | 0.04 | 2.00 | 8.00 |

| Sample count | Total estimated CNY | Mean CNY | P50 CNY | P95 CNY |
| ---: | ---: | ---: | ---: | ---: |
| 132 | 0.196565 | 0.001489 | 0.000641 | 0.005255 |

| Actual route | Sample count | Total estimated CNY | Mean CNY | P50 CNY | P95 CNY |
| --- | ---: | ---: | ---: | ---: | ---: |
| contact | 21 | 0.011858 | 0.000565 | 0.000551 | 0.000681 |
| direct | 20 | 0.010084 | 0.000504 | 0.000505 | 0.000641 |
| exact_product | 22 | 0.020472 | 0.000931 | 0.000676 | 0.001802 |
| fallback | 24 | 0.010297 | 0.000429 | 0.000000 | 0.002579 |
| knowledge | 21 | 0.046863 | 0.002232 | 0.001874 | 0.004709 |
| product_search | 24 | 0.096990 | 0.004041 | 0.003718 | 0.007651 |

## Estimated cost/conversation

Complete cost sample count: 4; mean CNY: 0.010015; P50 CNY: 0.007840.

| conversation_id | Complete | Estimated or partial CNY |
| --- | --- | ---: |
| conversation-capability-boundary | true | 0.006302 |
| conversation-product-selection | true | 0.009122 |
| conversation-product-to-contact | true | 0.007840 |
| conversation-solution-consultation | true | 0.016799 |

## Failures

Failed 0 of 132 attempts (rate 0.0000). Layers: {}. Codes: {}.

## Slowest requests

| Request ID | Route | Router type | Total ms | Model ms | Dominant stage | Ratio | Tool executions | Outcome |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| d18ce623f1674393820fada2ff8d4a16 | product_search | deterministic | 4031.0 | 3875.0 | model | 0.961 | 1 | success |
| 1c8751230bbf4328b676aeef51c0cdfc | fallback | llm | 3953.0 | 3953.0 | router | 1.000 | 0 | success |
| c2b5ff9ab67b49c4bbd1379aa33c60b3 | product_search | deterministic | 3688.0 | 3454.0 | model | 0.937 | 1 | success |
| d2041217bedd419fbcec0907a68d7fa2 | exact_product | deterministic | 3328.0 | 3297.0 | model | 0.991 | 1 | success |
| a8d65d4e4926410aa3549e11ca14b3bf | product_search | deterministic | 3203.0 | 3031.0 | model | 0.946 | 1 | success |
| 50ef71e1839f4afaac710d35431d94e6 | product_search | deterministic | 3156.0 | 2984.0 | model | 0.946 | 1 | success |
| c5aa382daef849aaad7ce4db19e54393 | product_search | deterministic | 3156.0 | 2985.0 | model | 0.946 | 1 | success |
| 6419faaa13754d10a1cb8ee0a3b78adb | product_search | llm | 3016.0 | 2845.0 | model | 0.943 | 1 | success |
| 7f099e924c1c4a679811151c07aa185d | product_search | deterministic | 2984.0 | 2703.0 | model | 0.906 | 1 | success |
| 6f3c20d3da7e40c495e4773356832e48 | knowledge | deterministic | 2953.0 | 2500.0 | model | 0.847 | 0 | success |

## Bottleneck conclusion

- Latency concentration: dominant measured stages were {"model": 108, "router": 16}; median dominant-stage ratio was 0.979.
- Slowest actual routes: highest route P50 was product_search (2235.0); highest route P95 was product_search (3688.0).
- Slow-row model diagnosis: 9 of 10 listed rows were model-dominant.
- Router-type comparison is descriptive, not causal: {"deterministic": {"count": 123, "mean": 1287.5853658536585, "p50": 1000.0, "p95": 2984.0}, "llm": {"count": 9, "mean": 1732.6666666666667, "p50": 1235.0, "p95": 3953.0}}.
- Retrieval and tool materiality are represented by their route-level non-null sample counts and latency percentiles in the stage table.
- Repeated product-search execution: 0 successful product-search requests executed more than one tool; 0 were in the listed slow rows.
- Highest-token route by average total tokens: product_search (3506.2); highest-cost route by mean estimated CNY: product_search (0.004041).

## Day 3 recommendation

- All 132 attempted requests succeeded; overall total latency was P50 1000 ms and P95 3156 ms.
- Product-search was the slowest actual route at P50 2235 ms and P95 3688 ms across 24 requests; its model latency was P50 1984 ms and P95 3454 ms.
- Nine of the ten slowest requests were model-dominant; model was dominant in 108 of 124 requests with a measurable dominant stage, with median dominant-stage ratio 0.979.
- Product-search tool latency was P95 438 ms across 23 non-null samples, and no successful product-search request executed more than one tool, so tool/agent execution is not the primary measured bottleneck.

## Limitations

This representative synthetic workload is not a production SLO or a universal conversation average. Router-type comparisons may have different route mixes. TTFT is not instrumented, so this evidence cannot establish a true-streaming TTFT improvement.

Repeated test cases may increase prompt cache reuse; observed estimated cost reflects the measured cache behavior of this representative synthetic workload.

C. Another specific measured bottleneck should be addressed
