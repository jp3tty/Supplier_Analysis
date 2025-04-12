import pandas as pd
import pymysql
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_db_config():
    """Get database configuration from environment variables"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'supply_chain'),
        'port': int(os.getenv('DB_PORT', 3306))
    }

def create_mysql_engine(db_config):
    """Create SQLAlchemy engine with error handling"""
    try:
        # First test the connection
        conn = pymysql.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            port=db_config['port'],
            ssl={'ca': None}  # Disable SSL for local development
        )
        
        # Create database if it doesn't exist
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_config['database']}")
        cursor.close()
        conn.close()
        
        # Create engine with the database
        connection_string = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        return create_engine(connection_string, connect_args={'ssl': {'ca': None}})
    except Exception as err:
        logger.error(f"MySQL Connection Error: {err}")
        logger.error("Please check your MySQL credentials and make sure the server is running")
        raise

def load_csv_to_mysql(csv_path, table_name, engine):
    """Load CSV file into MySQL table with encoding handling"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                logger.info(f"Trying to read {csv_path} with {encoding} encoding")
                df = pd.read_csv(csv_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Could not read {csv_path} with any of the attempted encodings")
        
        # Create table and load data
        logger.info(f"Loading data into table: {table_name}")
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace',
            index=False
        )
        
        logger.info(f"Successfully loaded {len(df)} rows into {table_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {str(e)}")
        return False

def main():
    try:
        # Get database configuration
        db_config = get_db_config()
        logger.info("Using database configuration:")
        logger.info(f"Host: {db_config['host']}")
        logger.info(f"User: {db_config['user']}")
        logger.info(f"Database: {db_config['database']}")
        logger.info(f"Port: {db_config['port']}")
        
        # Create MySQL engine
        engine = create_mysql_engine(db_config)
        
        # Define CSV files to load
        csv_files = {
            'DataCoSupplyChainDataset.csv': 'supply_chain_data',
            'DescriptionDataCoSupplyChain.csv': 'supply_chain_description',
            'tokenized_access_logs.csv': 'access_logs'
        }
        
        # Load each CSV file
        for csv_file, table_name in csv_files.items():
            csv_path = os.path.join('data', csv_file)
            if os.path.exists(csv_path):
                load_csv_to_mysql(csv_path, table_name, engine)
            else:
                logger.error(f"CSV file not found: {csv_path}")
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 