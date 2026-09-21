# Navigation model deliverable

This folder contains the three retained deployment candidates and the reports
needed to identify and validate them. All multi-Robotino results use 100
deterministic episodes with three Robotinos in the current factory geometry.

### Three-Robotino comparison

| Model | Fleet success | Any collision | Environment collision | Robotino collision | Timeout | Individual success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU corner fine-tuned (default) | **84%** | 4% | 2% | 2% | 12% | **94.3%** |
| Wide temporal CNN corner fine-tuned | 83% | 11% | 5% | 6% | **6%** | 92.3% |
| Wide temporal CNN (safety reference) | 75% | **1%** | **0%** | **1%** | 24% | 89.3% |

### Optimal Model Settings

| Model | Speed |
| --- | --- |
| GRU corner fine-tuned | 0.6 |
| Wide temporal CNN corner fine-tuned | 0.5 |
| Wide temporal CNN | 0.5 |

