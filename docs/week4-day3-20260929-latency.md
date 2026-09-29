# Week 4 Day 3 Streaming Closeout

## Result

Conclusion: **B**. At least one comparison gate failed.

## Explicit gates

- 50/50 baseline success: `True`
- 50/50 after success: `True`
- 100/100 success: `True`
- Exact route counts: `True`
- Complete baseline/after pairs: `True`
- Streaming mode verified: `True`
- Target routes match: `True`
- Client metrics complete: `True`
- Every measured success ended with `done` after the final render invariant: `True`
- product_search TTFT >=30% and >=500 ms: `True` (1172.0 ms, 0.42127965492451475)
- product_search client TTFT >=30% and >=500 ms: `True` (1171.9999999986612 ms, 0.4212796549241481)
- knowledge TTFT >=30% and >=500 ms: `False` (313.0 ms, 0.14619336758524054)
- knowledge client TTFT >=30% and >=500 ms: `False` (313.00000000192085 ms, 0.14619336758591514)
- product_search total-latency P50 <=15% regression: `True`
- knowledge total-latency P50 <=15% regression: `True`
- exact_product total-latency P50 <=15% regression: `True`
- contact total-latency P50 <=15% regression: `True`
- direct total-latency P50 <=15% regression: `True`
- All routes total-latency P50 <=15% regression: `True`
- All routes client total-latency P50 <=15% regression: `True`

## Route metrics

| Phase | Route | Success | Failure | Server TTFT P50/P95 ms | Client TTFT P50/P95 ms | Client total P50/P95 ms | Total P50/P95 ms | Model P50/P95 ms | Output tokens P50/P95 | Estimated cost CNY |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | product_search | 10 | 0 | 2782.0 / 4687.0 | 2781.9999999992433 / 4703.000000001339 | 2781.9999999992433 / 4703.000000001339 | 2782.0 / 4687.0 | 2016.0 / 3843.0 | 284.0 / 1122.0 | 0.020588 |
| baseline | knowledge | 10 | 0 | 2141.0 / 2797.0 | 2141.0000000032596 / 2796.999999998661 | 2141.0000000032596 / 2796.999999998661 | 2141.0 / 2797.0 | 1297.0 / 2031.0 | 35.0 / 378.0 | 0.009148 |
| baseline | exact_product | 10 | 0 | 1063.0 / 1265.0 | 1093.0000000007567 / 1264.999999999418 | 1093.0000000007567 / 1280.9999999954016 | 1063.0 / 1265.0 | 1063.0 / 1250.0 | 23.0 / 43.0 | 0.003545 |
| baseline | contact | 10 | 0 | 984.0 / 1281.0 | 984.0000000040163 / 1281.0000000026776 | 984.0000000040163 / 1296.9999999986612 | 984.0 / 1281.0 | 968.0 / 1265.0 | 28.0 / 76.0 | 0.003448 |
| baseline | direct | 10 | 0 | 1031.0 / 3391.0 | 1031.0000000026776 / 3391.0000000032596 | 1031.0000000026776 / 3406.0000000026776 | 1031.0 / 3406.0 | 1016.0 / 3360.0 | 21.0 / 39.0 | 0.002993 |
| after | product_search | 10 | 0 | 1610.0 / 1937.0 | 1610.000000000582 / 1953.0000000013388 | 2468.9999999973224 / 3985.000000000582 | 2469.0 / 3954.0 | 1656.0 / 3235.0 | 200.0 / 895.0 | 0.017001 |
| after | knowledge | 10 | 0 | 1828.0 / 2156.0 | 1828.0000000013388 / 2188.000000001921 | 2235.000000000582 / 3264.999999999418 | 2203.0 / 3250.0 | 1375.0 / 2532.0 | 37.0 / 450.0 | 0.011073 |
| after | exact_product | 10 | 0 | 766.0 / 1094.0 | 765.9999999959837 / 1110.000000000582 | 906.0000000026776 / 1281.9999999992433 | 906.0 / 1266.0 | 890.0 / 1250.0 | 26.0 / 37.0 | 0.003784 |
| after | contact | 10 | 0 | 781.0 / 1032.0 | 796.9999999986612 / 1031.9999999992433 | 921.9999999986612 / 1218.9999999973224 | 906.0 / 1219.0 | 891.0 / 1203.0 | 22.0 / 28.0 | 0.003132 |
| after | direct | 10 | 0 | 891.0 / 3594.0 | 906.0000000026776 / 3639.999999999418 | 1094.0000000045984 / 3655.9999999954016 | 1093.0 / 3625.0 | 1078.0 / 3593.0 | 20.0 / 38.0 | 0.002969 |

## Paired latency differences

Positive reductions mean streaming was faster; positive total/token changes mean streaming was larger or slower.

| Route | Matched | Server TTFT reduction P50/P95 ms | Client TTFT reduction P50/P95 ms | Total change P50/P95 ms | Output token change P50/P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| product_search | 10 | 1172.0 / 3062.0 | 1155.9999999954016 / 3078.000000001339 | -609.0 / 782.0 | -137.0 / 497.0 |
| knowledge | 10 | 313.0 / 1109.0 | 298.00000000250293 / 1093.9999999973224 | 203.0 / 453.0 | 3.0 / 213.0 |
| exact_product | 10 | 250.0 / 578.0 | 250.99999999656575 / 594.0000000045984 | -141.0 / 282.0 | 0.0 / 3.0 |
| contact | 10 | 265.0 / 562.0 | 249.00000000343425 / 578.0000000086147 | -187.0 / 375.0 | -6.0 / 0.0 |
| direct | 10 | 140.0 / 610.0 | 125.00000000727596 / 562.0000000053551 | 16.0 / 235.0 | -1.0 / 14.0 |

## Estimated cost and limitations

Estimated measured cost: CNY 0.077681.

Repeated test cases may increase prompt cache reuse; observed estimated cost reflects the measured cache behavior of this representative synthetic workload.

Client timing measures receipt of NDJSON events by this local HTTP test client; it is not browser rendering latency or a production SLO.
