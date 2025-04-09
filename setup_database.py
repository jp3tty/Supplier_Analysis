import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def setup_database():
    try:
        # Connect to MySQL server
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', '')
        )
        
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {os.getenv('DB_NAME', 'supply_chain')}")
        
        # Grant privileges to the user
        cursor.execute(f"GRANT ALL PRIVILEGES ON {os.getenv('DB_NAME', 'supply_chain')}.* TO '{os.getenv('DB_USER', 'root')}'@'localhost'")
        
        # Flush privileges
        cursor.execute("FLUSH PRIVILEGES")
        
        print("Database setup completed successfully!")
        
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    setup_database() 