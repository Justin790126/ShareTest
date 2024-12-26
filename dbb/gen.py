import sqlite3
import random

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()

# Create the table with 10 columns
cursor.execute('''
    CREATE TABLE IF NOT EXISTS my_table (
        col1 TEXT,
        col2 INTEGER,
        col3 REAL,
        col4 BLOB, 
        col5 TEXT,
        col6 INTEGER,
        col7 REAL,
        col8 BLOB,
        col9 TEXT,
        col10 INTEGER
    )
''')

# Generate random data for 87 rows
data = []
for _ in range(87):
    row = (
        f'row_{_}',  # Generate a simple text value
        random.randint(0, 100),  # Random integer between 0 and 100
        round(random.uniform(0, 10), 2),  # Random float between 0 and 10
        b'random_binary_data',  # Sample binary data (replace with your own generation)
        ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5)),  # Random 5-letter string
        random.randint(1, 50),
        round(random.uniform(1, 5), 2),
        b'more_random_binary',
        ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=8)),
        random.randint(10, 100)
    )
    data.append(row)

# Insert the data into the table
cursor.executemany("INSERT INTO my_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data)

# Commit the changes to the database
conn.commit()

# Close the connection
conn.close()