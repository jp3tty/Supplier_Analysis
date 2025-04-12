USE supply_chain;

SELECT
COUNT(DISTINCT `Order Id`) AS total_unique_orders
FROM supply_chain_data;

SELECT
COUNT(`Order Id`) AS total_entries
FROM supply_chain_data;

SELECT
	COUNT(*)- COUNT(DISTINCT `Order Id`) AS number_of_duplicates
FROM supply_chain_data;

SELECT
	`Order Id`,
    COUNT(*) AS occurrence_count
FROM supply_chain_data
GROUP BY `Order Id`
HAVING COUNT(*) > 1
ORDER BY occurrence_count DESC;

SELECT 
    occurrence_count,
    COUNT(*) as number_of_orders
FROM (
    SELECT 
        `Order Id`,
        COUNT(*) as occurrence_count
    FROM supply_chain_data
    GROUP BY `Order Id`
) as order_counts
GROUP BY occurrence_count
ORDER BY occurrence_count;