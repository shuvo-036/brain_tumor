import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from predict import predict

UPLOAD_FOLDER = "app/static/uploads"
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("index.html", error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template("index.html", error="No selected file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            res = predict(path)
            return render_template("index.html", result=res, img_path=path)
    return render_template("index.html")

if __name__ == "__main__":
    # Use 0.0.0.0:5000 to make available on your LAN
    app.run(host="0.0.0.0", port=5000, debug=True)
