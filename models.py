import sqlite3
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os

class User(UserMixin):
    def __init__(self, id, username, email, password_hash, created_at=None, last_login=None):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.created_at = created_at or datetime.now()
        self.last_login = last_login

    @staticmethod
    def create_table():
        conn = get_db_connection()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    @staticmethod
    def create(username, email, password):
        password_hash = generate_password_hash(password)
        conn = get_db_connection()
        try:
            cursor = conn.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            conn.commit()
            user_id = cursor.lastrowid
            return User.get_by_id(user_id)
        except sqlite3.IntegrityError as e:
            return None
        finally:
            conn.close()

    @staticmethod
    def get_by_id(user_id):
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        if user:
            return User(user['id'], user['username'], user['email'], 
                       user['password_hash'], user['created_at'], user['last_login'])
        return None

    @staticmethod
    def get_by_username(username):
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user:
            return User(user['id'], user['username'], user['email'], 
                       user['password_hash'], user['created_at'], user['last_login'])
        return None

    @staticmethod
    def get_by_email(email):
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        if user:
            return User(user['id'], user['username'], user['email'], 
                       user['password_hash'], user['created_at'], user['last_login'])
        return None

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def update_last_login(self):
        conn = get_db_connection()
        conn.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (self.id,))
        conn.commit()
        conn.close()

def get_db_connection():
    db_path = os.path.join(os.path.dirname(__file__), 'instance', 'users.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database
def init_db():
    User.create_table()