from flask import Flask, request, send_file, render_template, redirect
from threading import Thread
import os
import time
#import torch

# This is just a trial file showing how the changes wod be executed

app = Flask(__name__)

def modify_model(model_filename):
    # Simulate a long-running task by sleeping for 10 seconds
    time.sleep(10)
    # Call the bash script to modify the file
    os.system('bash new.sh')
    # Load the modified file
    # Rename the modified file with a unique name
    #os.rename('modified_model.pth', f'modified_model_{int(time.time())}.pth')

@app.route('/', methods=['GET', 'POST'])
def upload_model():
    if request.method == 'POST':
        # Get the uploaded file
        model_file = request.files['model']
        # Save the file
        model_file.save('model.pth')
        # Start the background task to modify the model
        Thread(target=modify_model, args=('model.pth',)).start()
        # Redirect to the page that shows the task is running
        return redirect('/download')
    return '''
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="model">
            <input type="submit" value="Upload">
        </form>
    '''

@app.route('/running')
def running():
    return render_template('running.html')

@app.route('/download')
def download():
    # Get the name of the most recently modified file
    files = os.listdir('.')
    files = [f for f in files if f.startswith('new')]
    files.sort()
    modified_model_filename = files[-1]
    # Return the modified file for download
    return send_file(modified_model_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
