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

-- Comparing the total_entries with unique and duplicate order id
SELECT
	COUNT(`Order Id`) AS total_entries,
    COUNT(DISTINCT `Order Id`) AS unique_orders,
    COUNT(*) - COUNT(DISTINCT `Order Id`) AS number_of_duplicates,
    GROUP_CONCAT(DISTINCT occurrence_count ORDER BY occurrence_count) AS unique_occurrence_counts
FROM supply_chain_data scd
CROSS JOIN(
	SELECT DISTINCT COUNT(*) AS occurrence_count
    FROM supply_chain_data
    GROUP BY `Order Id`)
AS counts;

-- Distribution of occurrence counts in a more readable format
SELECT 
    COUNT(*) as total_entries,
    COUNT(DISTINCT `Order Id`) as unique_orders,
    COUNT(*) - COUNT(DISTINCT `Order Id`) as number_of_duplicates,
    (
        SELECT GROUP_CONCAT(DISTINCT cnt ORDER BY cnt)
        FROM (
            SELECT COUNT(*) as cnt
            FROM supply_chain_data
            GROUP BY `Order Id`
        ) as counts
    ) as unique_occurrence_counts
FROM supply_chain_data;