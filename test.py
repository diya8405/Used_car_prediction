import sqlite3

con = sqlite3.connect("users.db")
cur = con.cursor()
cur.execute("SELECT * FROM users")
rows = cur.fetchall()
for row in rows:
    print(len(row), row)
con.close()