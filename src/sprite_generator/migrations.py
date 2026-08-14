"""
Sprite Generator — database migrations runner.
Reads all .sql files from the 'migrations' directory and runs them in order.
"""

import os
import psycopg2

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "migrations")

def run_migrations(db_url: str) -> None:
    """Apply all .sql files in the migrations directory in alphanumeric order."""
    if not db_url:
        print("[migrations] DB_URL not set — skipping migrations.", flush=True)
        return

    if not os.path.exists(MIGRATIONS_DIR):
        print(f"[migrations] Migrations directory '{MIGRATIONS_DIR}' not found.", flush=True)
        return

    # List .sql files, sort them to ensure order (e.g., 001_, 002_, ...)
    sql_files = sorted([f for f in os.listdir(MIGRATIONS_DIR) if f.endswith(".sql")])
    if not sql_files:
        print("[migrations] No .sql migration files found.", flush=True)
        return

    import time
    conn = None
    max_retries = 30
    retry_delay = 2

    for i in range(max_retries):
        try:
            conn = psycopg2.connect(db_url)
            print(f"[migrations] Connected to database on attempt {i+1}.", flush=True)
            break
        except Exception as e:
            if i < max_retries - 1:
                print(f"[migrations] Connection attempt {i+1}/{max_retries} failed ({e}). Retrying in {retry_delay}s…", flush=True)
                time.sleep(retry_delay)
            else:
                # Raise rather than return: a caller that starts anyway would serve
                # traffic against an unmigrated schema and look healthy doing it.
                raise RuntimeError(
                    f"[migrations] could not connect to database after {max_retries} attempts: {e}"
                ) from e

    try:
        # 1. Ensure the migrations tracking table exists
        with conn:
            with conn.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        filename TEXT PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
        
        # 2. Get the list of already applied migrations
        with conn:
            with conn.cursor() as cur:
                cur.execute('SELECT filename FROM schema_migrations')
                applied_migrations = {row[0] for row in cur.fetchall()}
        
        if applied_migrations:
            print(f"[migrations] Found {len(applied_migrations)} already applied migrations.", flush=True)

        # 3. Apply new migrations
        applied_any = False
        for filename in sql_files:
            if filename in applied_migrations:
                # Do NOT print anything for already applied migrations
                continue

            print(f"[migrations] Applying new migration: {filename}…", flush=True)
            filepath = os.path.join(MIGRATIONS_DIR, filename)
            with open(filepath, "r") as f:
                sql_content = f.read()

            # Execute migration and log it within the SAME transaction
            try:
                with conn:
                    with conn.cursor() as cur:
                        if sql_content.strip():
                            cur.execute(sql_content)
                        cur.execute(
                            'INSERT INTO schema_migrations (filename) VALUES (%s)',
                            (filename,)
                        )
                print(f"[migrations] Migration {filename} applied successfully.", flush=True)
                applied_any = True
            except Exception as e:
                # Stop on the first error to prevent out-of-order migrations, and
                # raise so the caller cannot proceed on a half-migrated schema.
                print(f"[migrations] ERROR while applying {filename}: {e}", flush=True)
                raise RuntimeError(f"[migrations] failed applying {filename}: {e}") from e

        conn.close()

        if applied_any:
            print("[migrations] Pending migrations completed.", flush=True)
        else:
            print("[migrations] Database is up to date.", flush=True)

    except Exception as exc:
        print(f"[migrations] ERROR — could not apply migrations: {exc}", flush=True)
        raise
    finally:
        if not conn.closed:
            conn.close()

if __name__ == "__main__":
    db_url = os.environ.get("DB_URL")
    if not db_url:
        print("[migrations] Error: DB_URL environment variable is required when running independently.", flush=True)
        exit(1)
    run_migrations(db_url)
