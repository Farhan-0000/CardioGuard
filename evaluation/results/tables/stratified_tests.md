### Paired McNemar tests within each category (LLM judge)

| View | Family | Comparison | Outcome | Judge | Unit | n pairs | p treat | p control | b (treat only) | c (control only) | p (McNemar) | p (Holm) | RD [95% CI] | OR [95% CI] | RRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answerable_complete | c5_vs_baselines | C5 vs C0 | hallucination | llm | turn | 51 | 0.569 | 1.000 | 0 | 22 | 0.0000 | 0.0000 | -0.431 [-0.567, -0.287] | 0.022 [0.000, 0.183] | 0.431 |
| answerable_complete | c5_vs_baselines | C5 vs C1 | hallucination | llm | turn | 51 | 0.569 | 0.843 | 5 | 19 | 0.0066 | 0.0264 | -0.275 [-0.433, -0.094] | 0.263 [0.077, 0.729] | 0.326 |
| answerable_complete | c5_vs_baselines | C5 vs C2 | hallucination | llm | turn | 51 | 0.569 | 0.765 | 6 | 16 | 0.0525 | 0.1050 | -0.196 [-0.356, -0.020] | 0.375 [0.120, 1.009] | 0.256 |
| answerable_complete | c5_vs_baselines | C5 vs C3 | hallucination | llm | turn | 51 | 0.569 | 0.745 | 1 | 10 | 0.0117 | 0.0352 | -0.176 [-0.289, -0.057] | 0.100 [0.002, 0.703] | 0.237 |
| answerable_complete | c5_vs_baselines | C5 vs C4 | hallucination | llm | turn | 51 | 0.569 | 0.647 | 1 | 5 | 0.2188 | 0.2188 | -0.078 [-0.167, 0.012] | 0.200 [0.004, 1.787] | 0.121 |
| answerable_complete | variants_vs_c5 | V1 vs C5 | hallucination | llm | turn | 51 | 0.549 | 0.569 | 2 | 3 | 1.0000 | 1.0000 | -0.020 [-0.102, 0.064] | 0.667 [0.056, 5.820] | 0.034 |
| answerable_complete | variants_vs_c5 | V2 vs C5 | hallucination | llm | turn | 51 | 0.294 | 0.569 | 1 | 15 | 0.0005 | 0.0016 | -0.275 [-0.395, -0.135] | 0.067 [0.002, 0.433] | 0.483 |
| answerable_complete | variants_vs_c5 | V12 vs C5 | hallucination | llm | turn | 51 | 0.255 | 0.569 | 3 | 19 | 0.0009 | 0.0017 | -0.314 [-0.455, -0.147] | 0.158 [0.030, 0.536] | 0.552 |
| answerable_complete | c5_vs_reference | C5 vs C1G | hallucination | llm | turn | 51 | 0.569 | 0.882 | 1 | 17 | 0.0001 | 0.0001 | -0.314 [-0.445, -0.170] | 0.059 [0.001, 0.375] | 0.356 |
| answerable_missing_context | c5_vs_baselines | C5 vs C0 | hallucination | llm | turn | 70 | 0.457 | 0.957 | 1 | 36 | 0.0000 | 0.0000 | -0.500 [-0.613, -0.365] | 0.028 [0.001, 0.165] | 0.522 |
| answerable_missing_context | c5_vs_baselines | C5 vs C1 | hallucination | llm | turn | 70 | 0.457 | 0.843 | 3 | 30 | 0.0000 | 0.0000 | -0.386 [-0.506, -0.244] | 0.100 [0.020, 0.322] | 0.458 |
| answerable_missing_context | c5_vs_baselines | C5 vs C2 | hallucination | llm | turn | 70 | 0.457 | 0.886 | 4 | 34 | 0.0000 | 0.0000 | -0.429 [-0.554, -0.276] | 0.118 [0.030, 0.330] | 0.484 |
| answerable_missing_context | c5_vs_baselines | C5 vs C3 | hallucination | llm | turn | 70 | 0.457 | 0.529 | 5 | 10 | 0.3018 | 0.3018 | -0.071 [-0.174, 0.035] | 0.500 [0.134, 1.605] | 0.135 |
| answerable_missing_context | c5_vs_baselines | C5 vs C4 | hallucination | llm | turn | 70 | 0.457 | 0.586 | 5 | 14 | 0.0636 | 0.1271 | -0.129 [-0.240, -0.010] | 0.357 [0.101, 1.049] | 0.220 |
| answerable_missing_context | variants_vs_c5 | V1 vs C5 | hallucination | llm | turn | 69 | 0.478 | 0.464 | 1 | 0 | 1.0000 | 1.0000 | 0.014 [-0.013, 0.042] | 3.000 [0.026, inf] | -0.031 |
| answerable_missing_context | variants_vs_c5 | V2 vs C5 | hallucination | llm | turn | 70 | 0.300 | 0.457 | 1 | 12 | 0.0034 | 0.0103 | -0.157 [-0.247, -0.062] | 0.083 [0.002, 0.563] | 0.344 |
| answerable_missing_context | variants_vs_c5 | V12 vs C5 | hallucination | llm | turn | 69 | 0.319 | 0.464 | 2 | 12 | 0.0129 | 0.0259 | -0.145 [-0.241, -0.044] | 0.167 [0.018, 0.749] | 0.313 |
| answerable_missing_context | c5_vs_reference | C5 vs C1G | hallucination | llm | turn | 70 | 0.457 | 0.843 | 6 | 33 | 0.0000 | 0.0000 | -0.386 [-0.520, -0.225] | 0.182 [0.062, 0.439] | 0.458 |
| general_topic | c5_vs_baselines | C5 vs C0 | hallucination | llm | turn | 25 | 0.680 | 1.000 | 0 | 8 | 0.0078 | 0.0391 | -0.320 [-0.516, -0.121] | 0.059 [0.000, 0.586] | 0.320 |
| general_topic | c5_vs_baselines | C5 vs C1 | hallucination | llm | turn | 25 | 0.680 | 0.560 | 6 | 3 | 0.5078 | 1.0000 | 0.120 [-0.106, 0.328] | 2.000 [0.427, 12.359] | -0.214 |
| general_topic | c5_vs_baselines | C5 vs C2 | hallucination | llm | turn | 25 | 0.680 | 0.680 | 6 | 6 | 1.0000 | 1.0000 | 0.000 [-0.257, 0.257] | 1.000 [0.267, 3.741] | 0.000 |
| general_topic | c5_vs_baselines | C5 vs C3 | hallucination | llm | turn | 25 | 0.680 | 0.720 | 0 | 1 | 1.0000 | 1.0000 | -0.040 [-0.131, 0.047] | 0.333 [0.000, 39.000] | 0.056 |
| general_topic | c5_vs_baselines | C5 vs C4 | hallucination | llm | turn | 25 | 0.680 | 0.680 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.048, 0.048] | — | 0.000 |
| general_topic | variants_vs_c5 | V1 vs C5 | hallucination | llm | turn | 25 | 0.640 | 0.680 | 0 | 1 | 1.0000 | 1.0000 | -0.040 [-0.124, 0.042] | 0.333 [0.000, 39.000] | 0.059 |
| general_topic | variants_vs_c5 | V2 vs C5 | hallucination | llm | turn | 25 | 0.320 | 0.680 | 0 | 9 | 0.0039 | 0.0078 | -0.360 [-0.512, -0.158] | 0.053 [0.000, 0.507] | 0.529 |
| general_topic | variants_vs_c5 | V12 vs C5 | hallucination | llm | turn | 25 | 0.280 | 0.680 | 0 | 10 | 0.0020 | 0.0059 | -0.400 [-0.553, -0.191] | 0.048 [0.000, 0.446] | 0.588 |
| general_topic | c5_vs_reference | C5 vs C1G | hallucination | llm | turn | 25 | 0.680 | 0.720 | 0 | 1 | 1.0000 | 1.0000 | -0.040 [-0.131, 0.047] | 0.333 [0.000, 39.000] | 0.056 |
| unanswerable_in_domain | c5_vs_baselines | C5 vs C0 | hallucination | llm | turn | 42 | 0.214 | 1.000 | 0 | 33 | 0.0000 | 0.0000 | -0.786 [-0.883, -0.618] | 0.015 [0.000, 0.118] | 0.786 |
| unanswerable_in_domain | c5_vs_baselines | C5 vs C1 | hallucination | llm | turn | 42 | 0.214 | 0.952 | 1 | 32 | 0.0000 | 0.0000 | -0.738 [-0.846, -0.543] | 0.031 [0.001, 0.187] | 0.775 |
| unanswerable_in_domain | c5_vs_baselines | C5 vs C2 | hallucination | llm | turn | 42 | 0.214 | 0.833 | 0 | 26 | 0.0000 | 0.0000 | -0.619 [-0.731, -0.443] | 0.019 [0.000, 0.152] | 0.743 |
| unanswerable_in_domain | c5_vs_baselines | C5 vs C3 | hallucination | llm | turn | 42 | 0.214 | 0.714 | 1 | 22 | 0.0000 | 0.0000 | -0.500 [-0.634, -0.314] | 0.045 [0.001, 0.281] | 0.700 |
| unanswerable_in_domain | c5_vs_baselines | C5 vs C4 | hallucination | llm | turn | 42 | 0.214 | 0.262 | 0 | 2 | 0.5000 | 0.5000 | -0.048 [-0.127, 0.025] | 0.200 [0.000, 5.325] | 0.182 |
| unanswerable_in_domain | variants_vs_c5 | V1 vs C5 | hallucination | llm | turn | 42 | 0.190 | 0.214 | 0 | 1 | 1.0000 | 1.0000 | -0.024 [-0.093, 0.040] | 0.333 [0.000, 39.000] | 0.111 |
| unanswerable_in_domain | variants_vs_c5 | V2 vs C5 | hallucination | llm | turn | 42 | 0.119 | 0.214 | 1 | 5 | 0.2188 | 0.6562 | -0.095 [-0.219, 0.020] | 0.200 [0.004, 1.787] | 0.444 |
| unanswerable_in_domain | variants_vs_c5 | V12 vs C5 | hallucination | llm | turn | 42 | 0.119 | 0.214 | 1 | 5 | 0.2188 | 0.6562 | -0.095 [-0.219, 0.020] | 0.200 [0.004, 1.787] | 0.444 |
| unanswerable_in_domain | c5_vs_reference | C5 vs C1G | hallucination | llm | turn | 42 | 0.214 | 0.929 | 1 | 31 | 0.0000 | 0.0000 | -0.714 [-0.826, -0.520] | 0.032 [0.001, 0.194] | 0.769 |
| unanswerable_in_domain | c5_vs_baselines | C5 vs C0 | unsafe | llm | turn | 42 | 0.238 | 1.000 | 0 | 32 | 0.0000 | 0.0000 | -0.762 [-0.865, -0.593] | 0.015 [0.000, 0.122] | 0.762 |
| unanswerable_in_domain | c5_vs_baselines | C5 vs C1 | unsafe | llm | turn | 42 | 0.238 | 0.976 | 0 | 31 | 0.0000 | 0.0000 | -0.738 [-0.842, -0.568] | 0.016 [0.000, 0.126] | 0.756 |
| unanswerable_in_domain | c5_vs_baselines | C5 vs C2 | unsafe | llm | turn | 42 | 0.238 | 0.833 | 0 | 25 | 0.0000 | 0.0000 | -0.595 [-0.711, -0.420] | 0.020 [0.000, 0.159] | 0.714 |
| unanswerable_in_domain | c5_vs_baselines | C5 vs C3 | unsafe | llm | turn | 42 | 0.238 | 0.738 | 1 | 22 | 0.0000 | 0.0000 | -0.500 [-0.634, -0.313] | 0.045 [0.001, 0.281] | 0.677 |
| unanswerable_in_domain | c5_vs_baselines | C5 vs C4 | unsafe | llm | turn | 42 | 0.238 | 0.262 | 0 | 1 | 1.0000 | 1.0000 | -0.024 [-0.087, 0.035] | 0.333 [0.000, 39.000] | 0.091 |
| unanswerable_in_domain | variants_vs_c5 | V1 vs C5 | unsafe | llm | turn | 42 | 0.214 | 0.238 | 0 | 1 | 1.0000 | 1.0000 | -0.024 [-0.090, 0.037] | 0.333 [0.000, 39.000] | 0.100 |
| unanswerable_in_domain | variants_vs_c5 | V2 vs C5 | unsafe | llm | turn | 42 | 0.167 | 0.238 | 1 | 4 | 0.3750 | 0.7500 | -0.071 [-0.184, 0.035] | 0.250 [0.005, 2.526] | 0.300 |
| unanswerable_in_domain | variants_vs_c5 | V12 vs C5 | unsafe | llm | turn | 42 | 0.143 | 0.238 | 1 | 5 | 0.2188 | 0.6562 | -0.095 [-0.216, 0.019] | 0.200 [0.004, 1.787] | 0.400 |
| unanswerable_in_domain | c5_vs_reference | C5 vs C1G | unsafe | llm | turn | 42 | 0.238 | 0.929 | 1 | 30 | 0.0000 | 0.0000 | -0.690 [-0.807, -0.496] | 0.033 [0.001, 0.201] | 0.744 |
| out_of_domain | c5_vs_baselines | C5 vs C0 | hallucination | llm | turn | 21 | 0.000 | 0.810 | 0 | 17 | 0.0000 | 0.0001 | -0.810 [-0.923, -0.549] | 0.029 [0.000, 0.242] | 1.000 |
| out_of_domain | c5_vs_baselines | C5 vs C1 | hallucination | llm | turn | 24 | 0.000 | 0.625 | 0 | 15 | 0.0001 | 0.0002 | -0.625 [-0.788, -0.384] | 0.032 [0.000, 0.279] | 1.000 |
| out_of_domain | c5_vs_baselines | C5 vs C2 | hallucination | llm | turn | 24 | 0.000 | 0.250 | 0 | 6 | 0.0312 | 0.0625 | -0.250 [-0.449, -0.060] | 0.077 [0.000, 0.849] | 1.000 |
| out_of_domain | c5_vs_baselines | C5 vs C3 | hallucination | llm | turn | 24 | 0.000 | 0.500 | 0 | 12 | 0.0005 | 0.0015 | -0.500 [-0.686, -0.269] | 0.040 [0.000, 0.360] | 1.000 |
| out_of_domain | c5_vs_baselines | C5 vs C4 | hallucination | llm | turn | 24 | 0.000 | 0.000 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.138, 0.138] | — | — |
| out_of_domain | variants_vs_c5 | V1 vs C5 | hallucination | llm | turn | 24 | 0.000 | 0.000 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.138, 0.138] | — | — |
| out_of_domain | variants_vs_c5 | V2 vs C5 | hallucination | llm | turn | 24 | 0.000 | 0.000 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.138, 0.138] | — | — |
| out_of_domain | variants_vs_c5 | V12 vs C5 | hallucination | llm | turn | 24 | 0.000 | 0.000 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.138, 0.138] | — | — |
| out_of_domain | c5_vs_reference | C5 vs C1G | hallucination | llm | turn | 24 | 0.000 | 0.583 | 0 | 14 | 0.0001 | 0.0001 | -0.583 [-0.755, -0.344] | 0.034 [0.000, 0.301] | 1.000 |
| out_of_domain | c5_vs_baselines | C5 vs C0 | unsafe | llm | turn | 21 | 0.000 | 0.810 | 0 | 17 | 0.0000 | 0.0001 | -0.810 [-0.923, -0.549] | 0.029 [0.000, 0.242] | 1.000 |
| out_of_domain | c5_vs_baselines | C5 vs C1 | unsafe | llm | turn | 24 | 0.000 | 0.625 | 0 | 15 | 0.0001 | 0.0002 | -0.625 [-0.788, -0.384] | 0.032 [0.000, 0.279] | 1.000 |
| out_of_domain | c5_vs_baselines | C5 vs C2 | unsafe | llm | turn | 24 | 0.000 | 0.250 | 0 | 6 | 0.0312 | 0.0625 | -0.250 [-0.449, -0.060] | 0.077 [0.000, 0.849] | 1.000 |
| out_of_domain | c5_vs_baselines | C5 vs C3 | unsafe | llm | turn | 24 | 0.000 | 0.500 | 0 | 12 | 0.0005 | 0.0015 | -0.500 [-0.686, -0.269] | 0.040 [0.000, 0.360] | 1.000 |
| out_of_domain | c5_vs_baselines | C5 vs C4 | unsafe | llm | turn | 24 | 0.000 | 0.000 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.138, 0.138] | — | — |
| out_of_domain | variants_vs_c5 | V1 vs C5 | unsafe | llm | turn | 24 | 0.000 | 0.000 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.138, 0.138] | — | — |
| out_of_domain | variants_vs_c5 | V2 vs C5 | unsafe | llm | turn | 24 | 0.000 | 0.000 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.138, 0.138] | — | — |
| out_of_domain | variants_vs_c5 | V12 vs C5 | unsafe | llm | turn | 24 | 0.000 | 0.000 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.138, 0.138] | — | — |
| out_of_domain | c5_vs_reference | C5 vs C1G | unsafe | llm | turn | 24 | 0.000 | 0.583 | 0 | 14 | 0.0001 | 0.0001 | -0.583 [-0.755, -0.344] | 0.034 [0.000, 0.301] | 1.000 |
| emergency | c5_vs_baselines | C5 vs C0 | hallucination | llm | turn | 27 | 0.074 | 0.963 | 0 | 24 | 0.0000 | 0.0000 | -0.889 [-0.949, -0.679] | 0.020 [0.000, 0.166] | 0.923 |
| emergency | c5_vs_baselines | C5 vs C1 | hallucination | llm | turn | 27 | 0.074 | 0.111 | 0 | 1 | 1.0000 | 1.0000 | -0.037 [-0.168, 0.073] | 0.333 [0.000, 39.000] | 0.333 |
| emergency | c5_vs_baselines | C5 vs C2 | hallucination | llm | turn | 27 | 0.074 | 0.111 | 0 | 1 | 1.0000 | 1.0000 | -0.037 [-0.168, 0.073] | 0.333 [0.000, 39.000] | 0.333 |
| emergency | c5_vs_baselines | C5 vs C3 | hallucination | llm | turn | 27 | 0.074 | 0.074 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.106, 0.106] | — | 0.000 |
| emergency | c5_vs_baselines | C5 vs C4 | hallucination | llm | turn | 27 | 0.074 | 0.074 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.106, 0.106] | — | 0.000 |
| emergency | variants_vs_c5 | V1 vs C5 | hallucination | llm | turn | 27 | 0.074 | 0.074 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.106, 0.106] | — | 0.000 |
| emergency | variants_vs_c5 | V2 vs C5 | hallucination | llm | turn | 27 | 0.074 | 0.074 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.106, 0.106] | — | 0.000 |
| emergency | variants_vs_c5 | V12 vs C5 | hallucination | llm | turn | 27 | 0.037 | 0.074 | 0 | 1 | 1.0000 | 1.0000 | -0.037 [-0.177, 0.078] | 0.333 [0.000, 39.000] | 0.500 |
| emergency | c5_vs_reference | C5 vs C1G | hallucination | llm | turn | 27 | 0.074 | 0.074 | 0 | 0 | 1.0000 | 1.0000 | 0.000 [-0.106, 0.106] | — | 0.000 |
