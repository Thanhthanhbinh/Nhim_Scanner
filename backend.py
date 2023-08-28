from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

def get_upload_no():
    if os.path.exists("uploads"):
        return len(os.listdir("uploads"))
    return 0

@app.route('/')
def hello():
    return render_template('index.html', image_no = get_upload_no())

@app.route('/test')
def test():
    return "testing"

@app.route('/file', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('file')
            
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        for f in files:
            f.save(os.path.join("uploads",secure_filename(f.filename)))
        return 'file uploaded successfully', 200
    if request.method == "GET":
        if os.path.isfile('scanned1.pdf'):
            return send_file(
        'scanned.pdf',
        mimetype='text/pdf',
        download_name='scanned.pdf',
        as_attachment=True
        )
        else:
            return "No file found", 404
    return 'Operation fail', 500
		
if __name__ == '__main__':
    app.run(debug = True)