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

    def update_password(self, new_password):
        new_hash = generate_password_hash(new_password)
        conn = get_db_connection()
        conn.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, self.id))
        conn.commit()
        conn.close()
        self.password_hash = new_hash

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
    FertilizerHistory.create_table()
    FarmingCostHistory.create_table()
    PricesDemandHistory.create_table()
    PestPredictionHistory.create_table()

import json

class FertilizerHistory:
    @staticmethod
    def create_table():
        conn = get_db_connection()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS fertilizer_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                fertilizer TEXT NOT NULL,
                confidence REAL,
                quantity TEXT,
                yield_val TEXT,
                cost REAL,
                growth_stage TEXT,
                purpose TEXT,
                sustainability_note TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()
        conn.close()

    @staticmethod
    def add_record(user_id, data):
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO fertilizer_history (
                user_id, timestamp, fertilizer, confidence, quantity, 
                yield_val, cost, growth_stage, purpose, sustainability_note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, 
            data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M')),
            data.get('fertilizer', ''),
            data.get('confidence', 0),
            data.get('quantity', ''),
            data.get('yield_val', ''),
            data.get('cost', 0),
            data.get('growth_stage', ''),
            data.get('purpose', ''),
            data.get('sustainability_note', '')
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def get_by_user(user_id, limit=20):
        conn = get_db_connection()
        history = conn.execute('''
            SELECT * FROM fertilizer_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit)).fetchall()
        conn.close()
        return [dict(row) for row in history]

class FarmingCostHistory:
    @staticmethod
    def create_table():
        conn = get_db_connection()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS farming_cost_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                predicted_cost REAL NOT NULL,
                total_input REAL NOT NULL,
                difference REAL NOT NULL,
                input_data TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()
        conn.close()

    @staticmethod
    def add_record(user_id, data):
        conn = get_db_connection()
        input_data_json = json.dumps(data.get('input_data', {}))
        conn.execute('''
            INSERT INTO farming_cost_history (
                user_id, timestamp, predicted_cost, total_input, difference, input_data
            )
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            data.get('predicted_cost', 0),
            data.get('total_input', 0),
            data.get('difference', 0),
            input_data_json
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def get_by_user(user_id, limit=500):
        conn = get_db_connection()
        history = conn.execute('''
            SELECT * FROM farming_cost_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit)).fetchall()
        conn.close()
        
        results = []
        for row in history:
            item = dict(row)
            item['input_data'] = json.loads(item['input_data'])
            results.append(item)
        return results


class PricesDemandHistory:
    @staticmethod
    def create_table():
        conn = get_db_connection()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prices_demand_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                avg_price REAL NOT NULL,
                avg_demand REAL NOT NULL,
                price_trend TEXT,
                demand_trend TEXT,
                recommended_action TEXT,
                recommended_date TEXT,
                recommended_price REAL,
                sensor_data TEXT,
                results_json TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()
        conn.close()

    @staticmethod
    def add_record(user_id, data):
        conn = get_db_connection()
        sensor_data = data.get('sensor_data')
        results_json = json.dumps(data.get('results', {}))
        sensor_data_json = json.dumps(sensor_data) if sensor_data else None

        conn.execute('''
            INSERT INTO prices_demand_history (
                user_id, timestamp, prediction_type, start_date, end_date,
                avg_price, avg_demand, price_trend, demand_trend,
                recommended_action, recommended_date, recommended_price,
                sensor_data, results_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            data.get('prediction_type', 'Standard'),
            data.get('start_date', ''),
            data.get('end_date', ''),
            data.get('avg_price', 0),
            data.get('avg_demand', 0),
            data.get('price_trend', ''),
            data.get('demand_trend', ''),
            data.get('recommended_action', ''),
            data.get('recommended_date'),
            data.get('recommended_price'),
            sensor_data_json,
            results_json
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def get_by_user(user_id, limit=100):
        conn = get_db_connection()
        history = conn.execute('''
            SELECT * FROM prices_demand_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit)).fetchall()
        conn.close()

        results = []
        for row in history:
            item = dict(row)
            item['sensor_data'] = json.loads(item['sensor_data']) if item.get('sensor_data') else None
            item['results_json'] = json.loads(item['results_json']) if item.get('results_json') else {}
            results.append(item)
        return results


class PestPredictionHistory:
    @staticmethod
    def create_table():
        conn = get_db_connection()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS pest_prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                predicted_pest TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                confidence REAL NOT NULL,
                recommended_action TEXT,
                input_data TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()
        conn.close()

    @staticmethod
    def add_record(user_id, data):
        conn = get_db_connection()
        input_data_json = json.dumps(data.get('input_data', {}))
        conn.execute('''
            INSERT INTO pest_prediction_history (
                user_id, timestamp, prediction_type, predicted_pest,
                risk_level, confidence, recommended_action, input_data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            data.get('prediction_type', 'Manual Entry'),
            data.get('predicted_pest', ''),
            data.get('risk_level', 'Low'),
            data.get('confidence', 0),
            data.get('recommended_action', ''),
            input_data_json
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def has_matching_record(user_id, prediction_type, predicted_pest, confidence, input_data):
        conn = get_db_connection()
        input_data_json = json.dumps(input_data or {})
        row = conn.execute('''
            SELECT id FROM pest_prediction_history
            WHERE user_id = ?
                AND prediction_type = ?
                AND predicted_pest = ?
                AND confidence = ?
                AND input_data = ?
            ORDER BY id DESC
            LIMIT 1
        ''', (user_id, prediction_type, predicted_pest, confidence, input_data_json)).fetchone()
        conn.close()
        return row is not None

    @staticmethod
    def get_by_user(user_id, limit=100):
        conn = get_db_connection()
        history = conn.execute('''
            SELECT * FROM pest_prediction_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit)).fetchall()
        conn.close()

        results = []
        for row in history:
            item = dict(row)
            item['input_data'] = json.loads(item['input_data']) if item.get('input_data') else {}
            results.append(item)
        return results