import sqlite3

def list_tables(db_file):
    """
    Lists all tables in the specified SQLite database.

    Args:
        db_file: Path to the SQLite database file.

    Returns:
        A list of table names.
    """

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        return tables

    except sqlite3.Error as e:
        print(f"Error listing tables: {e}")
        return []

    finally:
        if conn:
            conn.close()

def read_and_print_table(db_file, table_name):
    """
    Reads and prints rows from the specified table in the database.

    Args:
        db_file: Path to the SQLite database file.
        table_name: Name of the table to read.
    """

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        print(f"Table: {table_name}")
        for row in rows:
            print(row)
        print()

    except sqlite3.Error as e:
        print(f"Error reading table {table_name}: {e}")

    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    db_file = 'mydatabase.db'  # Replace with the actual path to your database file

    tables = list_tables(db_file)
    if tables:
        print("Tables in the database:")
        for table in tables:
            print(table)
        print()

        for table in tables:
            read_and_print_table(db_file, table)
    else:
        print("No tables found in the database.")