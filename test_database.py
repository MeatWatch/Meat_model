"""
Script untuk menguji koneksi database secara langsung.
Jalankan script ini terpisah untuk memastikan koneksi database berfungsi.
"""
from database import get_db_connection, close_connection

def test_database_connection():
    """Test koneksi ke database dan print hasil"""
    print("Mencoba membuat koneksi database...")
    
    try:
        conn = get_db_connection()
        
        if conn and conn.is_connected():
            print("✅ Koneksi berhasil! Database terhubung.")
            
            # Tes query sederhana
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            print("\nTabel dalam database:")
            
            tables = cursor.fetchall()
            for table in tables:
                print(f"- {table[0]}")
            
            # Pastikan tabel classifications ada
            cursor.execute("SHOW COLUMNS FROM classifications")
            print("\nKolom dalam tabel classifications:")
            
            columns = cursor.fetchall()
            for column in columns:
                print(f"- {column[0]} ({column[1]})")
            
            close_connection(conn, cursor)
        else:
            print("❌ Koneksi gagal! Database tidak terhubung.")
    
    except Exception as e:
        print(f"❌ Terjadi error saat koneksi database: {e}")

if __name__ == "__main__":
    test_database_connection()