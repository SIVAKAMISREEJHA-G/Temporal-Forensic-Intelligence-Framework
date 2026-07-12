import sqlite3, os, threading

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "tfif.db")
_local = threading.local()

def get_conn():
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS videos (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            filename     TEXT NOT NULL,
            orig_name    TEXT NOT NULL,
            upload_time  TEXT NOT NULL,
            duration     REAL,
            resolution   TEXT,
            fps          REAL,
            file_path    TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id        INTEGER REFERENCES videos(id),
            predicted_class TEXT,
            confidence      REAL,
            per_class_json  TEXT,
            timeline_json   TEXT,
            keyframes_json  TEXT,
            created_at      TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reports (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id      INTEGER REFERENCES videos(id),
            report_json   TEXT,
            pdf_path      TEXT,
            created_at    TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS job_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id   INTEGER REFERENCES videos(id),
            status     TEXT NOT NULL,
            stage      TEXT,
            message    TEXT,
            updated_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()
    print("Database initialised at", DB_PATH)
