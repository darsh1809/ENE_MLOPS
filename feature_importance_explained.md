# Feature Importance Examples

This document explains what the "Importance" values mean using real-world examples.

## The Concept

The model gave these scores:
1.  **AvgBasketValue**: `0.91` (91% Importance) - **The Decision Maker**
2.  **AvgBasketSize**: `0.05` (5% Importance) - **The Tie-Breaker**
3.  **Country**: `0.007` (0.7% Importance) - **Irrelevant**

## Example 1: The "Decision Maker" (AvgBasketValue)

Imagine you have two customers who are identical in every way, except for how much they spend per order.

| Customer | Country | Items per Order | **Avg Spend per Order** | **Predicted Segment** |
| :--- | :--- | :--- | :--- | :--- |
| **Alice** | UK | 10 | **$250.00** | **Champion** |
| **Bob** | UK | 10 | **$15.00** | **Lost / Low Value** |

*   **Observation:** Changing *only* the **Avg Spend** completely changed the result.
*   **Conclusion:** The model relies heavily on this number. That's why Importance is **0.91**.

## Example 2: The "Irrelevant" Feature (Country)

Now imagine two customers who spend the same amount, but live in different countries.

| Customer | Country | Items per Order | Avg Spend per Order | **Predicted Segment** |
| :--- | :--- | :--- | :--- | :--- |
| **Charlie** | **France** | 10 | $250.00 | **Champion** |
| **Dave** | **Germany** | 10 | $250.00 | **Champion** |

*   **Observation:** Changing the **Country** did *not* change the result. The model ignores it because high spenders are Champions regardless of where they live.
*   **Conclusion:** This feature doesn't help make decisions. That's why Importance is **0.007**.

## Example 3: The "Tie-Breaker" (AvgBasketSize)

Sometimes, if the Spend is in the middle, the number of items might help decide.

| Customer | Country | **Items per Order** | Avg Spend per Order | **Predicted Segment** |
| :--- | :--- | :--- | :--- | :--- |
| **Eve** | UK | **50 (Cheap items)** | $50.00 | **Potential Loyalist** |
| **Frank** | UK | **2 (Expensive items)** | $50.00 | **At Risk** |

*   **Observation:** Here, the number of items helped distinguish between two people who spent the same amount.
*   **Conclusion:** It helps a little bit, but not as much as the total money. That's why Importance is **0.05**.
