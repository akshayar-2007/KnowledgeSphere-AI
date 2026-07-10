from datetime import datetime, timezone
from functools import wraps
import json
import os
import re
import sqlite3

import numpy as np
import PyPDF2
from flask import Flask, abort, redirect, render_template, request, send_from_directory, session
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = "knowledge_portal_secret"
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    kw_model = KeyBERT(model)
except Exception:
    # Offline fallback: app still works with deterministic hashed embeddings.
    model = None
    kw_model = None

VALID_ROLES = ("student", "faculty", "administrator", "researcher")
ROLE_LABELS = {
    "student": "Student",
    "faculty": "Faculty",
    "administrator": "Administrator",
    "researcher": "Researcher",
}
AUDIENCE_OPTIONS = [
    ("all", "All Stakeholders"),
    ("students", "Students"),
    ("faculty", "Faculty"),
    ("researchers", "Researchers"),
    ("administrators", "Administrators"),
]


# ---------------- DATABASE ----------------
def get_db():
    conn = sqlite3.connect("database.db")
    return conn


def ensure_column(cur, table_name, column_name, definition):
    cur.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cur.fetchall()}
    if column_name not in existing:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document TEXT,
            domain TEXT,
            text TEXT,
            embedding TEXT,
            audience TEXT DEFAULT 'all',
            keywords TEXT DEFAULT '',
            uploaded_by TEXT DEFAULT '',
            created_at TEXT DEFAULT ''
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS domains(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS activity_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            action TEXT,
            details TEXT,
            created_at TEXT
        )
        """
    )

    # Migration support for old DBs.
    ensure_column(cur, "knowledge", "audience", "TEXT DEFAULT 'all'")
    ensure_column(cur, "knowledge", "keywords", "TEXT DEFAULT ''")
    ensure_column(cur, "knowledge", "uploaded_by", "TEXT DEFAULT ''")
    ensure_column(cur, "knowledge", "created_at", "TEXT DEFAULT ''")

    cur.execute("SELECT COUNT(*) FROM domains")
    if cur.fetchone()[0] == 0:
        cur.executemany(
            "INSERT INTO domains(name) VALUES (?)",
            [("Academics",), ("Administration",), ("Research",), ("General",)],
        )

    conn.commit()
    conn.close()


init_db()


# ---------------- HELPERS ----------------
def normalize_role(role):
    role = (role or "").strip().lower()
    legacy_map = {"admin": "administrator", "user": "student"}
    return legacy_map.get(role, role)


def role_label(role):
    return ROLE_LABELS.get(role, "Stakeholder")


def allowed_audiences_for_role(role):
    role = normalize_role(role)
    mapping = {
        "administrator": {"all", "students", "faculty", "researchers", "administrators"},
        "faculty": {"all", "students", "faculty", "researchers"},
        "researcher": {"all", "researchers"},
        "student": {"all", "students"},
    }
    return mapping.get(role, {"all"})


def can_upload(role):
    return normalize_role(role) in {"faculty", "administrator", "researcher"}


def can_manage_domains(role):
    return normalize_role(role) == "administrator"


def log_activity(username, action, details):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO activity_log(username,action,details,created_at) VALUES (?,?,?,?)",
            (username, action, details, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        conn.commit()
        conn.close()
    except Exception:
        # Logging should never break core workflow.
        pass


def login_required(route_fn):
    @wraps(route_fn)
    def wrapper(*args, **kwargs):
        if not session.get("username"):
            return redirect("/")
        return route_fn(*args, **kwargs)

    return wrapper


def roles_required(*roles):
    allowed_roles = {normalize_role(role) for role in roles}

    def decorator(route_fn):
        @wraps(route_fn)
        def wrapper(*args, **kwargs):
            current_role = normalize_role(session.get("role"))
            if not session.get("username"):
                return redirect("/")
            if current_role not in allowed_roles:
                return abort(403)
            return route_fn(*args, **kwargs)

        return wrapper

    return decorator


def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, "rb") as file_handle:
        reader = PyPDF2.PdfReader(file_handle)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_into_chunks(text, chunk_size=180, overlap=40):
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunk = " ".join(chunk_words).strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def hashed_embedding(text, dim=384):
    vector = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-zA-Z0-9]{2,}", text.lower())
    for token in tokens:
        vector[hash(token) % dim] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def encode_texts(texts):
    if not texts:
        return np.array([], dtype=np.float32)
    if model is not None:
        return model.encode(texts)
    return np.array([hashed_embedding(text) for text in texts], dtype=np.float32)


def auto_keywords(text):
    sample = " ".join(text.split()[:500])
    if not sample:
        return ""

    if kw_model is not None:
        try:
            extracted = kw_model.extract_keywords(
                sample,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=5,
            )
            return ", ".join([kw for kw, _ in extracted])
        except Exception:
            pass

    tokens = re.findall(r"[a-zA-Z]{4,}", sample.lower())
    unique = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
        if len(unique) == 5:
            break
    return ", ".join(unique)


def is_document_accessible(filename, role):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT audience FROM knowledge WHERE document=?", (filename,))
    audiences = [row[0] or "all" for row in cur.fetchall()]
    conn.close()
    if not audiences:
        return False
    allowed = allowed_audiences_for_role(role)
    return any(audience in allowed for audience in audiences)


def get_dashboard_stats(role):
    allowed = tuple(allowed_audiences_for_role(role))
    placeholders = ",".join(["?"] * len(allowed))

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        f"SELECT COUNT(DISTINCT document) FROM knowledge WHERE audience IN ({placeholders})",
        allowed,
    )
    doc_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM domains")
    domain_count = cur.fetchone()[0]

    cur.execute(
        f"""
        SELECT document, domain, MAX(created_at)
        FROM knowledge
        WHERE audience IN ({placeholders})
        GROUP BY document, domain
        ORDER BY MAX(created_at) DESC
        LIMIT 5
        """,
        allowed,
    )
    recent_docs = cur.fetchall()

    cur.execute("SELECT COUNT(*) FROM users")
    user_count = cur.fetchone()[0]

    conn.close()
    return {
        "doc_count": doc_count,
        "domain_count": domain_count,
        "recent_docs": recent_docs,
        "user_count": user_count,
    }


# ---------------- AUTH ----------------
@app.route("/")
def home():
    if session.get("username"):
        return redirect("/dashboard")
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        role = normalize_role(request.form["role"])

        if role not in VALID_ROLES:
            return render_template("register.html", error="Please choose a valid role.", roles=VALID_ROLES)

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username=?", (username,))
        existing = cur.fetchone()
        if existing:
            conn.close()
            return render_template("register.html", error="Username already exists.", roles=VALID_ROLES)

        cur.execute(
            "INSERT INTO users(username,password,role) VALUES (?,?,?)",
            (username, generate_password_hash(password), role),
        )
        conn.commit()
        conn.close()
        log_activity(username, "register", f"New {role} account created")
        return redirect("/")

    return render_template("register.html", roles=VALID_ROLES)


@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"].strip()
    password = request.form["password"]

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id,password,role FROM users WHERE username=?", (username,))
    user = cur.fetchone()

    if not user:
        conn.close()
        return render_template("index.html", error="Invalid username or password.")

    user_id, stored_password, role = user
    role = normalize_role(role)
    password_ok = False

    # Support all Werkzeug hash formats (pbkdf2/scrypt/etc.) and keep
    # backward compatibility for very old plaintext records.
    try:
        password_ok = check_password_hash(stored_password, password)
    except Exception:
        password_ok = False

    # Legacy plaintext record fallback.
    if not password_ok and ":" not in (stored_password or ""):
        password_ok = stored_password == password
        if password_ok:
            cur.execute(
                "UPDATE users SET password=?, role=? WHERE id=?",
                (generate_password_hash(password), role, user_id),
            )
            conn.commit()

    conn.close()

    if not password_ok:
        return render_template("index.html", error="Invalid username or password.")

    session["user_id"] = user_id
    session["username"] = username
    session["role"] = role
    log_activity(username, "login", f"{role} logged in")
    return redirect("/dashboard")


@app.route("/logout")
def logout():
    if session.get("username"):
        log_activity(session["username"], "logout", "User logged out")
    session.clear()
    return redirect("/")


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
@login_required
def dashboard():
    role = normalize_role(session.get("role"))
    stats = get_dashboard_stats(role)
    return render_template(
        "dashboard.html",
        role=role,
        role_label=role_label(role),
        username=session.get("username"),
        can_upload_docs=can_upload(role),
        can_manage_domain=can_manage_domains(role),
        stats=stats,
    )


@app.route("/admin")
@roles_required("administrator")
def admin_dashboard():
    return redirect("/dashboard")


@app.route("/user")
@login_required
def user():
    return redirect("/dashboard")


# ---------------- PDF UPLOAD ----------------
@app.route("/upload", methods=["GET", "POST"])
@roles_required("faculty", "administrator", "researcher")
def upload():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT name FROM domains ORDER BY name")
    domains = [row[0] for row in cur.fetchall()]
    conn.close()

    if request.method == "POST":
        files = request.files.getlist("pdf")
        domain = request.form["domain"]
        audience = request.form.get("audience", "all")
        manual_keywords = request.form.get("keywords", "").strip()
        username = session.get("username", "unknown")

        if audience not in {item[0] for item in AUDIENCE_OPTIONS}:
            return render_template(
                "upload.html",
                domains=domains,
                audience_options=AUDIENCE_OPTIONS,
                error="Invalid audience selected.",
            )

        saved_docs = 0
        conn = get_db()
        cur = conn.cursor()
        for file in files:
            if not file.filename:
                continue
            if not file.filename.lower().endswith(".pdf"):
                continue

            filename = os.path.basename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            text = extract_text_from_pdf(filepath)
            chunks = split_into_chunks(text)
            if not chunks:
                continue

            doc_keywords = manual_keywords or auto_keywords(text)
            embeddings = encode_texts(chunks)
            created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

            for idx, chunk in enumerate(chunks):
                cur.execute(
                    """
                    INSERT INTO knowledge(document,domain,text,embedding,audience,keywords,uploaded_by,created_at)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        filename,
                        domain,
                        chunk,
                        json.dumps(embeddings[idx].tolist()),
                        audience,
                        doc_keywords,
                        username,
                        created_at,
                    ),
                )

            saved_docs += 1
            log_activity(username, "upload", f"Uploaded {filename} in {domain}")

        conn.commit()
        conn.close()

        message = f"{saved_docs} document(s) uploaded successfully."
        return render_template(
            "upload.html",
            domains=domains,
            audience_options=AUDIENCE_OPTIONS,
            success=message,
        )

    return render_template("upload.html", domains=domains, audience_options=AUDIENCE_OPTIONS)


# ---------------- SEARCH ----------------
@app.route("/search", methods=["GET", "POST"])
@login_required
def search():
    role = normalize_role(session.get("role"))
    is_admin = role == "administrator"
    allowed_audiences = allowed_audiences_for_role(role)
    audience_choices = [item for item in AUDIENCE_OPTIONS if item[0] in allowed_audiences or item[0] == "all"]

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT name FROM domains ORDER BY name")
    domains = [row[0] for row in cur.fetchall()]
    conn.close()

    results = []
    selected_domain = "all"
    selected_audience = "all"
    search_query = ""

    if request.method == "POST":
        search_query = request.form["query"].strip()
        selected_domain = request.form.get("domain", "all")
        selected_audience = request.form.get("audience", "all") if is_admin else "all"

        allowed_tuple = tuple(allowed_audiences)
        placeholders = ",".join(["?"] * len(allowed_tuple))
        sql = (
            "SELECT document,domain,text,embedding,audience,keywords "
            f"FROM knowledge WHERE audience IN ({placeholders})"
        )
        params = list(allowed_tuple)

        if selected_domain != "all":
            sql += " AND domain=?"
            params.append(selected_domain)
        if is_admin and selected_audience != "all" and selected_audience in allowed_audiences:
            sql += " AND audience=?"
            params.append(selected_audience)

        conn = get_db()
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        conn.close()

        if search_query and rows:
            valid_rows = []
            embedding_vectors = []

            for doc, domain, text, embedding_json, audience, keywords in rows:
                try:
                    vector = np.array(json.loads(embedding_json), dtype=np.float32)
                except Exception:
                    vector = encode_texts([text])[0]

                valid_rows.append((doc, domain, text, audience, keywords or ""))
                embedding_vectors.append(vector)

            embeddings = np.array(embedding_vectors, dtype=np.float32)
            query_embedding = encode_texts([search_query])
            semantic_scores = cosine_similarity(query_embedding, embeddings)[0]
            query_terms = set(re.findall(r"[a-zA-Z]{3,}", search_query.lower()))

            ranked = []
            for row_data, semantic in zip(valid_rows, semantic_scores):
                _, _, text, _, keywords = row_data
                keyword_source = f"{keywords} {text[:300]}".lower()
                keyword_terms = set(re.findall(r"[a-zA-Z]{3,}", keyword_source))
                keyword_overlap = 0.0
                if query_terms:
                    keyword_overlap = len(query_terms.intersection(keyword_terms)) / len(query_terms)

                final_score = (0.85 * float(semantic)) + (0.15 * keyword_overlap)
                ranked.append((row_data, final_score))

            ranked.sort(key=lambda item: item[1], reverse=True)

            seen_documents = set()
            for (doc, domain, text, audience, keywords), score in ranked:
                if doc in seen_documents:
                    continue
                if score < 0.30:
                    continue

                results.append(
                    {
                        "document": doc,
                        "domain": domain,
                        "audience": audience,
                        "keywords": keywords,
                        "snippet": text[:240],
                        "score": int(score * 100),
                    }
                )
                seen_documents.add(doc)
                if len(results) == 5:
                    break

    return render_template(
        "search.html",
        results=results,
        role=role,
        domains=domains,
        selected_domain=selected_domain,
        selected_audience=selected_audience,
        audience_options=audience_choices,
        can_filter_audience=is_admin,
        search_query=search_query,
    )


# -------------- PREVIEW / DOWNLOAD -------------
@app.route("/preview/<filename>")
@login_required
def preview(filename):
    role = normalize_role(session.get("role"))
    if not is_document_accessible(filename, role):
        return abort(403)
    return send_from_directory("uploads", filename)


@app.route("/download/<filename>")
@login_required
def download(filename):
    role = normalize_role(session.get("role"))
    if not is_document_accessible(filename, role):
        return abort(403)
    return send_from_directory("uploads", filename, as_attachment=True)


# ----------- DOCUMENTS ------------
@app.route("/documents")
@login_required
def documents():
    role = normalize_role(session.get("role"))
    allowed = tuple(allowed_audiences_for_role(role))
    placeholders = ",".join(["?"] * len(allowed))

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT
            domain,
            document,
            audience,
            COALESCE(MAX(keywords), ''),
            COALESCE(MAX(created_at), '')
        FROM knowledge
        WHERE audience IN ({placeholders})
        GROUP BY domain, document, audience
        ORDER BY domain, document
        """,
        allowed,
    )
    rows = cur.fetchall()
    conn.close()

    data = {}
    categories = set()
    for domain, doc, audience, keywords, created_at in rows:
        categories.add(domain)
        if domain not in data:
            data[domain] = []
        data[domain].append(
            {
                "name": doc,
                "audience": audience,
                "keywords": keywords,
                "created_at": created_at,
            }
        )

    return render_template(
        "documents.html",
        data=data,
        categories=sorted(categories),
        role=role,
    )


# ---------------- DOMAINS ----------------
@app.route("/domains", methods=["GET", "POST"])
@roles_required("administrator")
def domains():
    message = None
    error = None
    conn = get_db()
    cur = conn.cursor()

    if request.method == "POST":
        name = request.form["name"].strip()
        if not name:
            error = "Domain name cannot be empty."
        else:
            try:
                cur.execute("INSERT INTO domains(name) VALUES (?)", (name,))
                conn.commit()
                message = "Domain added successfully."
                log_activity(session.get("username", "unknown"), "add_domain", name)
            except sqlite3.IntegrityError:
                error = "Domain already exists."

    cur.execute("SELECT name FROM domains ORDER BY name")
    rows = cur.fetchall()
    conn.close()

    return render_template("domains.html", domains=rows, message=message, error=error)


@app.errorhandler(403)
def forbidden(_error):
    return "You do not have permission to access this resource.", 403


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
