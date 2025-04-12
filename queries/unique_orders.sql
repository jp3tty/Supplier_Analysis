USE supply_chain;

SELECT
COUNT(DISTINCT `Order Id`) AS total_unique_orders
FROM supply_chain_data;