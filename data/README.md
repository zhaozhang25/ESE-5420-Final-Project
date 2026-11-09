# Data card

**Source:** [UCI Machine Learning Repository, dataset 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
**License:** CC BY 4.0
**File:** `raw/online_shoppers_intention.csv`
**SHA-256:** `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`

Do not edit the raw file. If it is lost, download it again from the UCI link above and check it against the SHA-256.

## Contents

12,330 sessions with 17 inputs and no missing values. The output is `Revenue` (TRUE for a purchase): 1,908 purchases, 15.47%.

| Columns | Type | Treatment |
|---|---|---|
| `Administrative`, `Informational`, `ProductRelated` | pages visited, by page type | scaled |
| `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` | time on those pages | scaled |
| `BounceRates`, `ExitRates`, `PageValues` | web-analytics metrics | scaled |
| `SpecialDay` | closeness to a special day | scaled |
| `Month`, `VisitorType`, `Weekend` | categorical | one-hot |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | integer codes with no order | one-hot |

## Splits

`splits/splits.csv` has one row per session used:

| Column | Meaning |
|---|---|
| `row` | position in the raw CSV, from 0, not counting the header |
| `role` | `dev` or `test` |
| `fold` | CV fold 0 to 4 for dev rows, -1 for test rows |

`splits/splits_keep_duplicates.csv` is the same for the sensitivity check that keeps duplicates.

## Known issues

- **`PageValues` may leak the label.** UCI defines it from pages visited before a transaction. On dev rows, 80.3% of buyers have `PageValues` > 0 versus 11.6% of non-buyers. This fits leakage but does not prove it, so results are reported with and without it.
- **125 exact duplicates.** All are non-purchase bounce sessions (`BounceRates` = `ExitRates` = 0.2), so they may be separate real visits rather than errors. There is no user ID to tell. The baseline config drops them, and `configs/keep_duplicates.yaml` keeps them.
- **Rare categories.** `Browser` 9 and `TrafficType` 12 and 17 have one row each, so the encoder ignores unseen levels.
- **No dates.** There is no year or date, and January and April are absent. Purchase rate varies by month from 1.6% (Feb) to 25.4% (Nov). Random folds say nothing about performance on future months.
