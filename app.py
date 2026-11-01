import streamlit as st
import joblib
import mysql.connector
import pandas as pd
import hashlib
import hmac
import secrets
import html as html_lib

from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent

# -------------------- Page configuration --------------------

st.set_page_config(
    page_title="Triage — Ticket Classification Console",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- Theme state --------------------

if "theme" not in st.session_state:
    st.session_state.theme = "dark"


def toggle_theme():
    st.session_state.theme = (
        "light" if st.session_state.theme == "dark" else "dark"
    )


def theme_button(container=None):
    """Render the dark/light toggle button."""
    target = container if container is not None else st
    is_dark = st.session_state.theme == "dark"
    target.button(
        "☀  Light mode" if is_dark else "🌙  Dark mode",
        key="theme_toggle_" + ("main" if container is not None else "top"),
        on_click=toggle_theme,
        use_container_width=True,
    )


PALETTES = {
    "dark": {
        "scheme": "dark",
        "bg": "#0B0E14",
        "panel": "#121722",
        "raised": "#171D2B",
        "border": "#232B3A",
        "text": "#E7EBF0",
        "muted": "#8B96A5",
        "blue": "#5B8CFF",
        "blue-hover": "#4778EC",
        "green": "#34D399",
        "yellow": "#FBBF24",
        "coral": "#FB8A65",
        "red": "#F0475A",
        "input": "#0E131D",
        "brand-bg": "#1C2A47",
        "brand-border": "#2B416B",
        "status-bg": "#102820",
        "status-border": "#1D4939",
        "empty-bg": "#0E131D",
        "empty-border": "#2A3447",
        "footer": "#687486",
        "row-alt": "#141A26",
    },
    "light": {
        "scheme": "light",
        "bg": "#F3F5F9",
        "panel": "#FFFFFF",
        "raised": "#F0F3F9",
        "border": "#D9DFEA",
        "text": "#141A26",
        "muted": "#5E6B7D",
        "blue": "#3B6FE8",
        "blue-hover": "#2F5CCB",
        "green": "#12946B",
        "yellow": "#B7791F",
        "coral": "#D9622B",
        "red": "#D12F42",
        "input": "#FFFFFF",
        "brand-bg": "#E4ECFF",
        "brand-border": "#BFD0FA",
        "status-bg": "#E3F6EE",
        "status-border": "#B5E5D0",
        "empty-bg": "#F7F9FC",
        "empty-border": "#C9D2E0",
        "footer": "#7A8699",
        "row-alt": "#F7F9FC",
    },
}


def build_theme_vars(theme):
    p = PALETTES[theme]
    lines = [f"  color-scheme: {p['scheme']};"]
    for key, value in p.items():
        if key != "scheme":
            lines.append(f"  --{key}: {value};")
    return ":root {\n" + "\n".join(lines) + "\n}"


# -------------------- Styles --------------------

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp { background: var(--bg); color: var(--text); }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.block-container {
    max-width: 1440px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
}
p, label, span, div { color: inherit; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

section[data-testid="stSidebar"] { background: var(--panel); }

div[data-testid="stWidgetLabel"] p,
label[data-baseweb="checkbox"] span {
  color: var(--text) !important;
}

div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] p {
  color: var(--muted) !important;
}

div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
  background: var(--input) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  font-family: 'Inter', sans-serif !important;
}

div[data-testid="stTextArea"] textarea::placeholder,
div[data-testid="stTextInput"] input::placeholder {
  color: var(--muted) !important;
  opacity: .8;
}

div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stTextInput"] input:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 1px var(--blue) !important;
}

div[data-testid="stForm"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
}

button[data-baseweb="tab"] { color: var(--muted) !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--text) !important; }

.stButton > button,
.stDownloadButton > button {
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--raised);
  color: var(--text);
  min-height: 42px;
  font-weight: 600;
  transition: all .15s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--blue);
    color: var(--blue);
}

div.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] button[kind="primary"] {
  background: var(--blue);
  color: white;
  border: 1px solid var(--blue);
}

div.stButton > button[kind="primary"]:hover {
    background: var(--blue-hover);
    color: white;
}

div[data-testid="stAlert"] {
  border-radius: 10px;
  background: var(--raised);
  border: 1px solid var(--border);
}

div[data-testid="stAlert"] p { color: var(--text) !important; }
hr { border-color: var(--border) !important; }

.topbar {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:16px;
  padding: 0 0 22px 0;
  border-bottom:1px solid var(--border);
  margin-bottom:26px;
}

.brand {
    display:flex;
    align-items:center;
    gap:12px;
}

.brand-icon {
  width:42px;
  height:42px;
  display:flex;
  align-items:center;
  justify-content:center;
  border-radius:12px;
  background:var(--brand-bg);
  color:var(--blue);
  font-size:22px;
  border:1px solid var(--brand-border);
}

.brand-title {
  font-family:'Space Grotesk',sans-serif;
  font-size:22px;
  font-weight:700;
  letter-spacing:-.5px;
}

.brand-sub {
    color:var(--muted);
    font-size:12px;
    margin-top:3px;
}

.status {
  color:var(--green);
  background:var(--status-bg);
  border:1px solid var(--status-border);
  border-radius:999px;
  padding:7px 11px;
  font-size:11px;
  font-weight:700;
  letter-spacing:.5px;
  white-space:nowrap;
}

.panel {
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:16px;
  padding:22px;
  min-height:100%;
}

.panel-kicker {
  color:var(--muted);
  text-transform:uppercase;
  font-size:10px;
  letter-spacing:1.5px;
  font-weight:700;
}

.panel-title {
  font-family:'Space Grotesk',sans-serif;
  font-size:20px;
  font-weight:600;
  margin-top:6px;
  margin-bottom:5px;
}

.panel-desc {
  color:var(--muted);
  font-size:12px;
  margin-bottom:20px;
}

.sample-label {
    color:var(--muted);
    font-size:11px;
    margin:10px 0 8px;
}

.result-card {
  background:var(--raised);
  border:1px solid var(--border);
  border-radius:12px;
  padding:15px 16px;
  margin-bottom:12px;
}

.result-label {
  color:var(--muted);
  font-size:11px;
  text-transform:uppercase;
  letter-spacing:1px;
  font-weight:700;
}

.result-value {
  font-family:'Space Grotesk',sans-serif;
  font-size:20px;
  font-weight:600;
  margin-top:7px;
  overflow-wrap:anywhere;
}

.result-dot {
  display:inline-block;
  width:8px;
  height:8px;
  border-radius:50%;
  margin-right:7px;
}

.small-muted {
    color:var(--muted);
    font-size:11px;
}

.history-item {
  border-bottom:1px solid var(--border);
  padding:12px 0;
}

.history-text {
    color:var(--text);
    font-size:12px;
    line-height:1.5;
}

.history-meta {
  color:var(--muted);
  font-family:'JetBrains Mono',monospace;
  font-size:10px;
  margin-top:6px;
}

.empty-state {
  text-align:center;
  padding:38px 15px;
  border:1px dashed var(--empty-border);
  border-radius:12px;
  color:var(--muted);
  background:var(--empty-bg);
}

.empty-icon {
    font-size:26px;
    margin-bottom:8px;
}

.table-wrap {
  max-height: 440px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--panel);
  margin-bottom: 14px;
}

table.tt {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

table.tt th {
  position: sticky;
  top: 0;
  background: var(--raised);
  color: var(--muted);
  text-transform: uppercase;
  font-size: 10px;
  letter-spacing: 1px;
  text-align: left;
  padding: 11px 14px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}

table.tt td {
  padding: 11px 14px;
  border-bottom: 1px solid var(--border);
  color: var(--text);
  vertical-align: top;
}

table.tt tr:nth-child(even) td { background: var(--row-alt); }
table.tt tr:last-child td { border-bottom: none; }
table.tt td.mono {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  white-space: nowrap;
}

.pill {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 10px;
  font-weight: 700;
  border: 1px solid var(--border);
  color: var(--muted);
}

.pill.on {
  color: var(--green);
  border-color: var(--status-border);
  background: var(--status-bg);
}

@media (max-width: 700px) {
  .block-container {
      padding-left: 1rem;
      padding-right: 1rem;
  }
  .panel { padding:16px; }
  .brand-title { font-size:19px; }
}
"""

st.markdown(
    "<style>\n"
    + build_theme_vars(st.session_state.theme)
    + "\n"
    + BASE_CSS
    + "\n</style>",
    unsafe_allow_html=True,
)


# -------------------- Database connection --------------------

def db_connection():

    if "mysql" not in st.secrets:
        raise RuntimeError("Missing [mysql] settings in Streamlit Secrets.")

    cfg = st.secrets["mysql"]

    required = ["host", "port", "user", "password", "database"]

    missing = [key for key in required if key not in cfg]

    if missing:
        raise RuntimeError(
            "Missing database settings: " + ", ".join(missing)
        )

    args = {
        "host": cfg["host"],
        "port": int(cfg["port"]),
        "user": cfg["user"],
        "password": cfg["password"],
        "database": cfg["database"],
        "connection_timeout": 20,
    }

    if cfg.get("ssl_ca"):

        ca_path = Path(cfg["ssl_ca"])

        if not ca_path.is_absolute():
            ca_path = BASE_DIR / ca_path

        if not ca_path.exists():
            raise RuntimeError(
                f"SSL certificate file not found: {ca_path}"
            )

        args["ssl_ca"] = str(ca_path)
        args["ssl_verify_cert"] = True

    return mysql.connector.connect(**args)


# -------------------- Database initialization --------------------

def init_database():

    conn = db_connection()

    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(150) NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                user_id BIGINT NULL,
                ticket_description TEXT NOT NULL,
                department VARCHAR(150),
                priority VARCHAR(100),
                topic VARCHAR(150),
                is_public BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration: add is_public to an existing tickets table.
        # Existing tickets stay private (FALSE) by default.
        try:
            cur.execute(
                "ALTER TABLE tickets ADD COLUMN is_public BOOLEAN DEFAULT FALSE"
            )
        except mysql.connector.Error as e:
            if e.errno != 1060:  # 1060 = duplicate column, already migrated
                raise

        conn.commit()
        cur.close()

    finally:
        conn.close()


# -------------------- Password security --------------------

def hash_password(password):

    salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        310000
    )

    return salt.hex() + ":" + password_hash.hex()


def verify_password(password, stored_hash):

    try:
        salt_hex, hash_hex = stored_hash.split(":")

        salt = bytes.fromhex(salt_hex)

        check_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            310000
        )

        return hmac.compare_digest(
            check_hash.hex(),
            hash_hex
        )

    except (ValueError, TypeError):
        return False


# -------------------- Registration --------------------

def register_user(name, email, password):

    conn = db_connection()

    try:
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO users (name, email, password_hash)
            VALUES (%s, %s, %s)
            """,
            (
                name.strip(),
                email.strip().lower(),
                hash_password(password)
            )
        )

        conn.commit()

        return cur.lastrowid

    finally:
        conn.close()


# -------------------- Login authentication --------------------

def authenticate_user(email, password):

    conn = db_connection()

    try:
        cur = conn.cursor(dictionary=True)

        cur.execute(
            """
            SELECT user_id, name, email, password_hash
            FROM users
            WHERE email = %s
            """,
            (email.strip().lower(),)
        )

        user = cur.fetchone()

        if user and verify_password(
            password,
            user["password_hash"]
        ):

            return {
                "user_id": user["user_id"],
                "name": user["name"],
                "email": user["email"]
            }

        return None

    finally:
        conn.close()


# -------------------- Ticket storage --------------------

def save_ticket(
    user_id,
    description,
    department,
    priority,
    topic,
    is_public=False
):

    conn = db_connection()

    try:
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO tickets
            (
                user_id,
                ticket_description,
                department,
                priority,
                topic,
                is_public
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                description,
                department,
                priority,
                topic,
                bool(is_public)
            )
        )

        conn.commit()

        ticket_id = cur.lastrowid

        cur.close()

        return ticket_id

    finally:
        conn.close()


# -------------------- Fetch user's tickets --------------------

def fetch_tickets(user_id, limit=50):

    conn = db_connection()

    try:
        cur = conn.cursor(dictionary=True)

        cur.execute(
            """
            SELECT
                id,
                ticket_description,
                department,
                priority,
                topic,
                is_public,
                created_at
            FROM tickets
            WHERE user_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (user_id, int(limit))
        )

        rows = cur.fetchall()

        cur.close()

        return pd.DataFrame(
            rows,
            columns=[
                "id",
                "ticket_description",
                "department",
                "priority",
                "topic",
                "is_public",
                "created_at"
            ]
        )

    finally:
        conn.close()


# -------------------- Fetch community tickets --------------------

def fetch_community_tickets(user_id, limit=100):

    conn = db_connection()

    try:
        cur = conn.cursor(dictionary=True)

        # Only public tickets from OTHER users.
        # Descriptions and email addresses are never selected.
        cur.execute(
            """
            SELECT
                t.id,
                u.name AS username,
                t.department,
                t.priority,
                t.topic,
                t.created_at
            FROM tickets t
            INNER JOIN users u
                ON t.user_id = u.user_id
            WHERE t.is_public = TRUE
              AND t.user_id != %s
            ORDER BY t.created_at DESC, t.id DESC
            LIMIT %s
            """,
            (user_id, int(limit))
        )

        rows = cur.fetchall()

        cur.close()

        return pd.DataFrame(
            rows,
            columns=[
                "id",
                "username",
                "department",
                "priority",
                "topic",
                "created_at"
            ]
        )

    finally:
        conn.close()


# -------------------- ML Model loading --------------------

@st.cache_resource
def load_models():

    department = joblib.load(
        BASE_DIR / "artifacts" / "model_department.pkl"
    )

    priority = joblib.load(
        BASE_DIR / "artifacts" / "model_priority.pkl"
    )

    topic = joblib.load(
        BASE_DIR / "artifacts_data3" / "model_topic.pkl"
    )

    return department, priority, topic


def predict_with_confidence(model, text):

    prediction = model.predict([text])[0]

    confidence = None

    try:
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba([text])

            if (
                probabilities is not None
                and len(probabilities)
                and len(probabilities[0])
            ):
                confidence = float(max(probabilities[0]))

    except Exception:
        confidence = None

    return str(prediction), confidence


def priority_color(value):

    value = str(value).lower()

    if "critical" in value or "urgent" in value:
        return "var(--red)"

    if "high" in value:
        return "var(--coral)"

    if "medium" in value or "normal" in value:
        return "var(--yellow)"

    if "low" in value:
        return "var(--green)"

    return "var(--blue)"


# -------------------- HTML table helper (theme-aware) --------------------

def esc(value):
    return html_lib.escape("" if value is None else str(value))


def render_table(df, columns, priority_col=None, mono_cols=(), pill_col=None):
    """
    Render a DataFrame as a themed HTML table.
    columns: ordered list of (dataframe_column, header_label)
    """

    head = "".join(f"<th>{esc(label)}</th>" for _, label in columns)

    body = ""

    for _, row in df.iterrows():

        cells = ""

        for col, _label in columns:

            value = row[col]

            if col == priority_col:
                cells += (
                    f'<td><span class="result-dot" '
                    f'style="background:{priority_color(value)}"></span>'
                    f'{esc(value)}</td>'
                )

            elif col == pill_col:
                on = bool(value)
                cells += (
                    f'<td><span class="pill {"on" if on else ""}">'
                    f'{"PUBLIC" if on else "PRIVATE"}</span></td>'
                )

            elif col in mono_cols:
                cells += f'<td class="mono">{esc(value)}</td>'

            else:
                cells += f"<td>{esc(value)}</td>"

        body += f"<tr>{cells}</tr>"

    return (
        '<div class="table-wrap"><table class="tt">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


# -------------------- Database status --------------------

if "db_ready" not in st.session_state:

    try:
        init_database()

        st.session_state.db_ready = True
        st.session_state.db_error = ""

    except Exception as e:

        st.session_state.db_ready = False
        st.session_state.db_error = str(e)


# -------------------- Session state --------------------

if "user" not in st.session_state:
    st.session_state.user = None

if "history" not in st.session_state:
    st.session_state.history = []

if "ticket_input" not in st.session_state:
    st.session_state.ticket_input = ""

if "latest" not in st.session_state:
    st.session_state.latest = None


# -------------------- Login / Registration page --------------------

if not st.session_state.db_ready:

    st.error(
        "Database is not connected: "
        + st.session_state.get("db_error", "Unknown database error")
    )

    st.stop()


if st.session_state.user is None:

    _, theme_col = st.columns([5, 1])

    with theme_col:
        theme_button(theme_col)

    st.markdown("""
    <div style="max-width:600px;margin:30px auto 25px;text-align:center;">
        <div style="font-size:38px;">✳</div>
        <div class="panel-title" style="font-size:30px;">
            Welcome to Triage
        </div>
        <div class="panel-desc">
            AI-powered ticket classification and management.
            Login or create an account to continue.
        </div>
    </div>
    """, unsafe_allow_html=True)

    login_tab, register_tab = st.tabs(
        ["Login", "Create Account"]
    )

    # -------------------- Login --------------------

    with login_tab:

        with st.form("login_form"):

            email = st.text_input(
                "Email address",
                key="login_email"
            )

            password = st.text_input(
                "Password",
                type="password",
                key="login_password"
            )

            login_clicked = st.form_submit_button(
                "Login",
                type="primary",
                use_container_width=True
            )

            if login_clicked:

                if not email.strip() or not password:

                    st.warning(
                        "Please enter your email and password."
                    )

                else:

                    try:

                        user = authenticate_user(
                            email,
                            password
                        )

                        if user:

                            st.session_state.user = user
                            st.session_state.history = []
                            st.session_state.latest = None

                            st.rerun()

                        else:

                            st.error(
                                "Invalid email or password."
                            )

                    except Exception as e:

                        st.error(f"Login failed: {e}")

    # -------------------- Registration --------------------

    with register_tab:

        with st.form("register_form"):

            name = st.text_input(
                "Full name",
                key="register_name"
            )

            email = st.text_input(
                "Email address",
                key="register_email"
            )

            password = st.text_input(
                "Create password",
                type="password",
                key="register_password"
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="confirm_password"
            )

            register_clicked = st.form_submit_button(
                "Create Account",
                type="primary",
                use_container_width=True
            )

            if register_clicked:

                if (
                    not name.strip()
                    or not email.strip()
                    or not password
                    or not confirm_password
                ):

                    st.warning(
                        "Please fill in all fields."
                    )

                elif (
                    "@" not in email
                    or "." not in email.split("@")[-1]
                ):

                    st.warning(
                        "Please enter a valid email address."
                    )

                elif len(password) < 8:

                    st.warning(
                        "Password must contain at least 8 characters."
                    )

                elif password != confirm_password:

                    st.error(
                        "Passwords do not match."
                    )

                else:

                    try:

                        register_user(
                            name,
                            email,
                            password
                        )

                        st.success(
                            "Account created successfully! "
                            "Please login using your credentials."
                        )

                    except mysql.connector.Error as e:

                        if e.errno == 1062:

                            st.error(
                                "An account with this email already exists."
                            )

                        else:

                            st.error(
                                f"Registration failed: {e}"
                            )

                    except Exception as e:

                        st.error(
                            f"Registration failed: {e}"
                        )

    st.stop()


# -------------------- Logged-in user --------------------

user_id = st.session_state.user["user_id"]

user_col, theme_col, logout_col = st.columns([5, 1.2, 1])

with user_col:

    st.markdown(
        f"**Welcome, {esc(st.session_state.user['name'])}**"
    )

with theme_col:

    theme_button(theme_col)

with logout_col:

    if st.button(
        "Logout",
        use_container_width=True
    ):

        st.session_state.user = None
        st.session_state.history = []
        st.session_state.latest = None
        st.session_state.ticket_input = ""

        st.rerun()


# -------------------- Header --------------------

st.markdown("""
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">✳</div>
    <div>
      <div class="brand-title">Triage</div>
      <div class="brand-sub">
        Ticket classification console
      </div>
    </div>
  </div>
  <div class="status">
    ● &nbsp; CLASSIFIER READY
  </div>
</div>
""", unsafe_allow_html=True)


# -------------------- Sample tickets --------------------

samples = {
    "Printer issue":
        "The office printer keeps jamming every time I try to print documents. Please help resolve this.",

    "Production outage":
        "The production checkout server is down and customers cannot complete purchases. This is urgent.",

    "Monitor request":
        "I would like to request an additional monitor for my desk. No rush, whenever available.",

    "Password locked":
        "I am locked out of my account after several failed password attempts and need access restored.",
}


# -------------------- Main classification interface --------------------

left, right = st.columns(
    [1.04, 0.96],
    gap="large"
)


# -------------------- Left: Ticket intake --------------------

with left:

    st.markdown("""
    <div class="panel">
      <div class="panel-kicker">01 / Intake</div>
      <div class="panel-title">Ticket details</div>
      <div class="panel-desc">
        Paste a support request below or start from a sample ticket.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sample-label">TRY A SAMPLE TICKET</div>',
        unsafe_allow_html=True
    )

    sample_cols = st.columns(2)

    sample_names = list(samples.keys())

    for idx, col in enumerate(sample_cols):

        with col:

            if st.button(
                sample_names[idx],
                use_container_width=True,
                key=f"sample_{idx}"
            ):

                st.session_state.ticket_input = (
                    samples[sample_names[idx]]
                )

                st.rerun()

    sample_cols2 = st.columns(2)

    for idx, col in enumerate(sample_cols2, start=2):

        with col:

            if st.button(
                sample_names[idx],
                use_container_width=True,
                key=f"sample_{idx}"
            ):

                st.session_state.ticket_input = (
                    samples[sample_names[idx]]
                )

                st.rerun()

    ticket = st.text_area(
        "Ticket description",
        key="ticket_input",
        height=210,
        placeholder="Describe the issue, request, or incident in detail...",
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div class="small-muted" style="text-align:right;margin-top:-8px;margin-bottom:12px;">{len(ticket)} characters</div>',
        unsafe_allow_html=True,
    )

    share_ticket = st.checkbox(
        "Share this ticket with other users",
        value=False,
        help=(
            "Other users can see your username, ticket ID, department, "
            "priority and topic. Your ticket description remains private."
        ),
    )

    classify = st.button(
        "✦  Classify ticket",
        type="primary",
        use_container_width=True
    )

    st.markdown("""
    <div class="small-muted" style="margin-top:12px;">
      Predictions are generated by your saved machine-learning models.
    </div>
    """, unsafe_allow_html=True)


    # -------------------- Classification + DB save --------------------

    if classify:

        if not ticket.strip():

            st.warning(
                "Please enter a ticket description."
            )

        else:

            try:

                department_model, priority_model, topic_model = load_models()

                dept, dept_conf = predict_with_confidence(
                    department_model,
                    ticket.strip()
                )

                pri, pri_conf = predict_with_confidence(
                    priority_model,
                    ticket.strip()
                )

                top, top_conf = predict_with_confidence(
                    topic_model,
                    ticket.strip()
                )

                # Save ticket against the logged-in user.

                ticket_id = save_ticket(
                    user_id,
                    ticket.strip(),
                    dept,
                    pri,
                    top,
                    share_ticket
                )

                result = {
                    "id": ticket_id,
                    "ticket": ticket.strip(),

                    "department": dept,
                    "department_confidence": dept_conf,

                    "priority": pri,
                    "priority_confidence": pri_conf,

                    "topic": top,
                    "topic_confidence": top_conf,

                    "shared": share_ticket,

                    "time": datetime.now().strftime("%H:%M:%S"),
                }

                st.session_state.latest = result

                st.session_state.history.insert(
                    0,
                    result
                )

                st.session_state.history = (
                    st.session_state.history[:8]
                )

                st.success(
                    f"Ticket #{ticket_id} classified and saved successfully!"
                    + (" It is shared with the community." if share_ticket else "")
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# -------------------- Right: Classification results --------------------

with right:

    st.markdown("""
    <div class="panel">
      <div class="panel-kicker">02 / Readout</div>
      <div class="panel-title">Classification results</div>
      <div class="panel-desc">
        Model output for the submitted support ticket.
      </div>
    </div>
    """, unsafe_allow_html=True)

    latest = st.session_state.latest

    if latest is None:

        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">⌁</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:16px;color:var(--text);margin-bottom:6px;">
            Awaiting ticket
          </div>
          <div style="font-size:12px;">
            Your classification results will appear here.
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(
            f'<div class="result-card"><div class="result-label">Department</div>'
            f'<div class="result-value"><span class="result-dot" style="background:var(--blue)"></span>{esc(latest["department"])}</div></div>',
            unsafe_allow_html=True,
        )

        pri_color = priority_color(
            latest["priority"]
        )

        st.markdown(
            f'<div class="result-card"><div class="result-label">Priority</div>'
            f'<div class="result-value"><span class="result-dot" style="background:{pri_color}"></span>{esc(latest["priority"])}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="result-card"><div class="result-label">Topic</div>'
            f'<div class="result-value"><span class="result-dot" style="background:var(--green)"></span>{esc(latest["topic"])}</div></div>',
            unsafe_allow_html=True,
        )

        confidences = [
            ("Department", latest["department_confidence"]),
            ("Priority", latest["priority_confidence"]),
            ("Topic", latest["topic_confidence"]),
        ]

        available = [
            (name, conf)
            for name, conf in confidences
            if conf is not None
        ]

        if available:

            st.markdown("#### Model confidence")

            for name, conf in available:

                st.markdown(
                    f'<div class="small-muted" style="margin-top:10px;margin-bottom:5px;">{name} <span style="float:right;color:var(--text);">{conf:.0%}</span></div>',
                    unsafe_allow_html=True
                )

                st.progress(
                    max(0.0, min(1.0, conf))
                )

            st.caption(
                "Confidence is shown only when the model provides "
                "prediction probabilities; it is not a guarantee of correctness."
            )

        else:

            st.caption(
                "Your models returned predictions successfully. "
                "Probability-based confidence is not exposed by one or more "
                "models, so no confidence score is displayed."
            )


# -------------------- Session queue --------------------

st.markdown(
    "<div style='height:18px'></div>",
    unsafe_allow_html=True
)

st.markdown("""
<div class="panel-kicker">03 / Session queue</div>
<div class="panel-title" style="font-size:20px;margin-bottom:5px;">
    Recent classifications
</div>
<div class="panel-desc">
    Your latest predictions in this session.
</div>
""", unsafe_allow_html=True)


if not st.session_state.history:

    st.markdown("""
    <div class="empty-state">
      <div style="font-size:13px;">
        No classifications yet. Submit a ticket to populate the queue.
      </div>
    </div>
    """, unsafe_allow_html=True)

else:

    for item in st.session_state.history:

        preview = esc(item["ticket"])

        if len(preview) > 145:
            preview = preview[:142] + "..."

        shared_tag = " &nbsp;·&nbsp; SHARED" if item.get("shared") else ""

        st.markdown(
            f'<div class="history-item"><div class="history-text">{preview}</div>'
            f'<div class="history-meta">#{item.get("id", "-")} &nbsp;·&nbsp; {item["time"]} &nbsp;·&nbsp; DEPT: {esc(item["department"])} '
            f'&nbsp;·&nbsp; PRIORITY: {esc(item["priority"])} &nbsp;·&nbsp; TOPIC: {esc(item["topic"])}{shared_tag}</div></div>',
            unsafe_allow_html=True,
        )


# -------------------- Permanent user-specific database history --------------------

st.markdown("---")

st.markdown("""
<div class="panel-kicker">04 / My database records</div>
<div class="panel-title">My saved tickets</div>
<div class="panel-desc">
    Your tickets permanently stored in TiDB Cloud. Only you can see the full description.
</div>
""", unsafe_allow_html=True)


try:

    saved_df = fetch_tickets(user_id)

    if saved_df.empty:

        st.info(
            "You have not submitted any tickets yet."
        )

    else:

        st.caption(
            f"You have {len(saved_df)} recent saved tickets."
        )

        st.markdown(
            render_table(
                saved_df,
                columns=[
                    ("id", "Ticket ID"),
                    ("ticket_description", "Description"),
                    ("department", "Department"),
                    ("priority", "Priority"),
                    ("topic", "Topic"),
                    ("is_public", "Visibility"),
                    ("created_at", "Created At"),
                ],
                priority_col="priority",
                pill_col="is_public",
                mono_cols=("id", "created_at"),
            ),
            unsafe_allow_html=True,
        )

        csv_df = saved_df.copy()
        csv_df["is_public"] = csv_df["is_public"].astype(bool)

        st.download_button(
            "Download my ticket history (CSV)",
            csv_df.to_csv(index=False).encode("utf-8"),
            "my_ticket_history.csv",
            "text/csv",
            use_container_width=True
        )

except Exception as e:

    st.error(
        f"Could not load your saved tickets: {e}"
    )


# -------------------- Community Tickets --------------------

st.markdown("---")

st.markdown("""
<div class="panel-kicker">05 / Community</div>
<div class="panel-title">Community Tickets</div>
<div class="panel-desc">
    Public tickets shared by other registered users.
    Descriptions and email addresses are never shown.
</div>
""", unsafe_allow_html=True)


try:

    community_df = fetch_community_tickets(user_id)

    if community_df.empty:

        st.info(
            "No public tickets from other users yet."
        )

    else:

        st.caption(
            f"{len(community_df)} public tickets shared by other users."
        )

        st.markdown(
            render_table(
                community_df,
                columns=[
                    ("id", "Ticket ID"),
                    ("username", "Raised By"),
                    ("department", "Department"),
                    ("priority", "Priority"),
                    ("topic", "Topic"),
                    ("created_at", "Shared At"),
                ],
                priority_col="priority",
                mono_cols=("id", "created_at"),
            ),
            unsafe_allow_html=True,
        )

except Exception as e:

    st.error(
        f"Could not load community tickets: {e}"
    )


# -------------------- Footer --------------------

st.markdown("""
<div style="border-top:1px solid var(--border);margin-top:28px;padding-top:14px;color:var(--footer);font-size:11px;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;">
  <span>TRIAGE · ML-POWERED TICKET CLASSIFICATION</span>
  <span>Department · Priority · Topic</span>
</div>
""", unsafe_allow_html=True)
