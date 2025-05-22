import mysql.connector
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'shinkansen.proxy.rlwy.net'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'yrhRErbEZgstknSqblQByYfuzLdISlTF'),
    'database': os.getenv('DB_NAME', 'railway'),
    'port': int(os.getenv('DB_PORT', 3000))
}

# For debug purposes - prints the current configuration
print(f"Database Configuration:")
print(f"  Host: {DB_CONFIG['host']}")
print(f"  Port: {DB_CONFIG['port']}")
print(f"  Database: {DB_CONFIG['database']}")
print(f"  User: {DB_CONFIG['user']}")
print(f"  Password: {'*' * len(DB_CONFIG['password'])}")

def get_db_connection():
    """
    Create and return a new database connection.
    """
    retry_count = 0
    max_retries = 2
    retry_delay = 1  # seconds
    
    while retry_count < max_retries:
        try:
            print(f"Connecting to database (attempt {retry_count + 1}/{max_retries})...")
            connection = mysql.connector.connect(**DB_CONFIG)
            
            if connection.is_connected():
                print("Database connection established successfully!")
                return connection
            
        except mysql.connector.Error as err:
            retry_count += 1
            print(f"Database connection error: {err}")
            
            if retry_count < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Failed to connect to database after multiple attempts.")
                raise
    
    return None

def close_connection(connection, cursor=None):
    """
    Safely close database connection and cursor.
    """
    if cursor:
        cursor.close()
        print("Database cursor closed.")
    
    if connection:
        connection.close()
        print("Database connection closed.")