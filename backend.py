from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('ui.html')

@app.route('/test')
def test():
    return "testing"

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'
		
if __name__ == '__main__':
   app.run(debug = True)