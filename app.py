from flask import Flask, request, render_template, redirect, url_for, session
import pickle
import os
import json
import numpy as np

app = Flask(__name__, static_folder="static")
app.secret_key = "your_secret_key_here"

# === Load KMeans model, scaler, and feature config ===
model = pickle.load(open("models/kmeans_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
career_config = pickle.load(open("models/career_config.pkl", "rb"))

feature_order = career_config["feature_order"]
cluster_labels = career_config.get("cluster_labels", {})  # dict like {0: "Likely Science Track", ...}

USERS_FILE = "users/users.json"

# === Routes ===

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        data = request.form
        username = data.get("username")
        password = data.get("password")

        os.makedirs("users", exist_ok=True)

        if not os.path.exists(USERS_FILE) or os.path.getsize(USERS_FILE) == 0:
            with open(USERS_FILE, "w") as f:
                json.dump({}, f)

        with open(USERS_FILE, "r") as f:
            try:
                users = json.load(f)
            except json.JSONDecodeError:
                users = {}

        if username in users:
            return "Username already exists", 400

        users[username] = {"password": password}

        with open(USERS_FILE, "w") as f:
            json.dump(users, f)

        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.form
        username = data.get("username")
        password = data.get("password")

        with open(USERS_FILE, "r") as f:
            users = json.load(f)

        if username not in users or users[username]["password"] != password:
            return "Invalid credentials", 401

        session["student_name"] = username
        return redirect(url_for("dashboard"))

    return render_template("login.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    student_name = session.get("student_name")
    prediction = None

    if request.method == "POST":
        form_key_map = {
            "ENGLISH": "english",
            "MATHEMATICS": "math",
            "SOCIAL STUDIES": "social",
            "AGRIC SCIENCE": "agric",
            "PHE": "phe",
            "BASIC TECH": "btech",
            "COMPUTER": "computer",
            "BUSINESS STUDIES": "business",
            "IRS/CRS": "religious_studies",
            "CCA": "cca",
            "YORUBA": "yoruba"
        }

        scores = []
        for subject in feature_order:
            form_key = form_key_map.get(subject)
            try:
                val = float(request.form.get(form_key, 0))
            except (TypeError, ValueError):
                val = 0.0
            scores.append(val)

        input_data = np.array(scores).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        cluster = model.predict(input_scaled)[0]

        label = cluster_labels.get(cluster, f"Cluster-{cluster}")
        prediction = f"You belong to Cluster-{cluster}: {label.split(':')[-1].strip().capitalize()}"

    return render_template("dashboard.html", prediction=prediction, student_name=student_name)

    student_name = session.get("student_name")
    prediction = None

    if request.method == "POST":
        form_key_map = {
            "ENGLISH": "english",
            "MATHEMATICS": "math",
            "SOCIAL STUDIES": "social",
            "AGRIC SCIENCE": "agric",
            "PHE": "phe",
            "BASIC TECH": "btech",
            "COMPUTER": "computer",
            "BUSINESS STUDIES": "business",
            "IRS/CRS": "religious_studies",
            "CCA": "cca",
            "YORUBA": "yoruba"
        }

        # Collect and map scores in training order
        scores = []
        for subject in feature_order:
            form_key = form_key_map.get(subject)
            try:
                val = float(request.form.get(form_key, 0))
            except (TypeError, ValueError):
                val = 0.0
            scores.append(val)

        # Standardize input and predict cluster
        input_data = np.array(scores).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        predicted_cluster = model.predict(input_scaled)[0]

        # Build multiline output with all clusters
        prediction_lines = []
        for i in sorted(cluster_labels):
            label = cluster_labels[i]
            if i == predicted_cluster:
                prediction_lines.append(f"<strong>You belong to Cluster-{i+1}: {label}</strong>")
            else:
                prediction_lines.append(f"You belong to Cluster-{i+1}: {label}")

        prediction = "<br>".join(prediction_lines)

    return render_template("dashboard.html", prediction=prediction, student_name=student_name)

    student_name = session.get("student_name")
    prediction = None

    if request.method == "POST":
        form_key_map = {
            "ENGLISH": "english",
            "MATHEMATICS": "math",
            "SOCIAL STUDIES": "social",
            "AGRIC SCIENCE": "agric",
            "PHE": "phe",
            "BASIC TECH": "btech",
            "COMPUTER": "computer",
            "BUSINESS STUDIES": "business",
            "IRS/CRS": "religious_studies",
            "CCA": "cca",
            "YORUBA": "yoruba"
        }

        # Collect and map scores in training order
        scores = []
        for subject in feature_order:
            form_key = form_key_map.get(subject)
            try:
                val = float(request.form.get(form_key, 0))
            except (TypeError, ValueError):
                val = 0.0
            scores.append(val)

        # Standardize input and predict cluster
        input_data = np.array(scores).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        cluster = model.predict(input_scaled)[0]

        # Map cluster to human-readable label
        cluster_label = cluster_labels.get(cluster, f"Cluster {cluster}")
        prediction = f"🔍 You belong to: {cluster_label}"

    return render_template("dashboard.html", prediction=prediction, student_name=student_name)

# === Run server ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
