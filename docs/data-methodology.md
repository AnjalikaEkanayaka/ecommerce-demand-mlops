# Data and forecasting methodology

## Target

The target is recorded order-item count per purchase date, aggregated across
the marketplace. Each unique (order_id, order_item_id) pair counts as one item.
The stored column is named total_units_sold.

Final delivery status is not used. Recorded items from canceled orders are
included. This measures recorded order intake, not fulfilled sales, returns,
or demand lost because of stock shortages.

## Reporting period

Processing requires explicit start and end dates within the observed date
range. Choose a complete reporting period and exclude partial days.

Missing dates are rejected by default. Use --fill-missing-days only when the
source is complete and absent days genuinely mean zero recorded items.

Timestamps are treated as local, timezone-naive values. No timezone conversion
is performed.

## Features

The forecast is rolling one-day-ahead demand.

For date D, features use calendar information and demand observed through D-1:

- Day of week, month, and day of month.
- Demand one day earlier.
- Demand seven days earlier.
- Mean demand over the previous seven days.

Same-day realized average price is excluded because it is unavailable before
that day's transactions occur. The first seven observations supply historical
context and are not training examples.

## Validation

Training and retraining share the same feature function and chronological
split. The latest 14 featured days are reserved for validation.

Later validation-day features may use actual demand from earlier validation
days. This represents forecasts refreshed daily, not a 14-day forecast issued
at one fixed date.

The offline demonstration assumes daily counts are complete before the next
forecast. The static dataset does not establish production data-arrival timing.

Candidate comparison, safe promotion, and a separate final reporting holdout
remain unfinished.

## Compatibility

Model input changed from seven features to six. Old price-based models cannot
be reused. New training will be needed after promotion safeguards are finished.