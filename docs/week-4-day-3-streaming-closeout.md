# Week 4 Day 3 Streaming Closeout

## Result

Conclusion: **B**. Streaming semantics are safe, but at least one performance gate failed.

Production deployment: **buffered**; the experimental streaming implementation remains covered but is not injected.

## Explicit gates

- 18/18 baseline success: `True`
- 18/18 after success: `True`
- 36/36 success: `True`
- Exact route counts: `True`
- Every measured success ended with `done` after the final render invariant: `True`
- product_search TTFT >=30% and >=500 ms: `True` (781.0 ms, 0.4235357917570499)
- knowledge TTFT >=30% and >=500 ms: `True` (1125.0 ms, 0.51440329218107)
- product_search total-latency P50 <=15% regression: `False`
- knowledge total-latency P50 <=15% regression: `True`
- direct total-latency P50 <=15% regression: `True`
- All routes total-latency P50 <=15% regression: `False`

## Route metrics

| Phase | Route | Success | Failure | TTFT P50/P95 ms | Buffering saved P50/P95 ms | Total P50/P95 ms | Model P50/P95 ms | Output tokens P50/P95 | Estimated cost CNY |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | product_search | 6 | 0 | 1844.0 / 2360.0 | 0.0 / 16.0 | 1844.0 / 2375.0 | 1703.0 / 2172.0 | 173.0 / 434.0 | 0.014573 |
| baseline | knowledge | 6 | 0 | 2187.0 / 3922.0 | 0.0 / 16.0 | 2187.0 / 3922.0 | 1593.0 / 2359.0 | 59.0 / 326.0 | 0.011149 |
| baseline | direct | 6 | 0 | 984.0 / 1250.0 | 0.0 / 0.0 | 984.0 / 1250.0 | 984.0 / 1250.0 | 16.0 / 39.0 | 0.003192 |
| after | product_search | 6 | 0 | 1063.0 / 1484.0 | 672.0 / 1750.0 | 2156.0 / 3031.0 | 1797.0 / 2891.0 | 94.0 / 498.0 | 0.014541 |
| after | knowledge | 6 | 0 | 1062.0 / 3047.0 | 391.0 / 1641.0 | 1469.0 / 3328.0 | 1219.0 / 2375.0 | 70.0 / 363.0 | 0.010021 |
| after | direct | 6 | 0 | 781.0 / 1828.0 | 219.0 / 1375.0 | 1016.0 / 3203.0 | 1000.0 / 3203.0 | 21.0 / 37.0 | 0.003136 |

## Estimated cost and limitations

Estimated measured cost: CNY 0.056611.

Repeated test cases may increase prompt cache reuse; observed estimated cost reflects the measured cache behavior of this representative synthetic workload.

This is synthetic server-side TTFT, not browser rendering latency or a production SLO.
