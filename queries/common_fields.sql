-- Find common column names between tables
SELECT 
    COLUMN_NAME,
    TABLE_NAME
FROM 
    INFORMATION_SCHEMA.COLUMNS
WHERE 
    TABLE_SCHEMA = 'supply_chain'
    AND TABLE_NAME IN ('supply_chain_data', 'supply_chain_description', 'access_logs')
ORDER BY 
    COLUMN_NAME;