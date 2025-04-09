import mysql.connector
from dotenv import load_dotenv
import os
import getpass

# Load environment variables
load_dotenv()

def setup_mysql():
    try:
        # Get root password
        root_password = getpass.getpass("Enter MySQL root password: ")
        
        # Connect to MySQL as root
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password=root_password
        )
        
        cursor = conn.cursor()
        
        # Create new user
        cursor.execute(f"CREATE USER IF NOT EXISTS '{os.getenv('DB_USER')}'@'localhost' IDENTIFIED BY '{os.getenv('DB_PASSWORD')}'")
        
        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {os.getenv('DB_NAME')}")
        
        # Grant privileges
        cursor.execute(f"GRANT ALL PRIVILEGES ON {os.getenv('DB_NAME')}.* TO '{os.getenv('DB_USER')}'@'localhost'")
        
        # Flush privileges
        cursor.execute("FLUSH PRIVILEGES")
        
        print("MySQL setup completed successfully!")
        print(f"Created user: {os.getenv('DB_USER')}")
        print(f"Created database: {os.getenv('DB_NAME')}")
        
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    setup_mysql() 