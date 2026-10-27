# Heuristic tuning: top 5 of 32 configs (train seeds)

| rank | config | J_overall | J stable | J high_churn | success | p95_all_ms | pings/h | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | f=2,po=0,rs=1,ps=0,tg=all | 0.9752 | 0.9761 ± 0.0026 | 0.9742 ± 0.0032 | 0.9995 | 395.2 | 23.5 | global best, privileged stable |
| 2 | f=2,po=1,rs=1,ps=0,tg=all | 0.9752 | 0.9761 ± 0.0026 | 0.9742 ± 0.0032 | 0.9995 | 395.2 | 23.5 |  |
| 3 | f=2,po=0,rs=1,ps=0,tg=top_two | 0.9751 | 0.9760 ± 0.0026 | 0.9742 ± 0.0032 | 0.9995 | 394.7 | 22.9 | privileged high_churn |
| 4 | f=2,po=1,rs=1,ps=0,tg=top_two | 0.9751 | 0.9760 ± 0.0026 | 0.9742 ± 0.0032 | 0.9995 | 394.7 | 22.9 |  |
| 5 | f=2,po=0,rs=1,ps=1,tg=top_two | 0.9751 | 0.9759 ± 0.0026 | 0.9742 ± 0.0032 | 0.9995 | 395.5 | 23.5 |  |
| 22 | f=2,po=1,rs=0,ps=1,tg=all | 0.9740 | 0.9743 ± 0.0034 | 0.9738 ± 0.0033 | 0.9995 | 400.3 | 33.5 | default |
