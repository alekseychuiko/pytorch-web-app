from flask import render_template, request, send_from_directory, flash, url_for
from flask import current_app as app
from .model import predict, supported_image_type
from . import photos


class DataObject():
    pass


def checkFileType(f: str):
    return f.split('.')[-1] in ['jpg', 'jpeg', 'png', 'gif', 'svg', 'bmp']


def cleanString(v: str):
    out_str = v
    delm = ['_', '-', '.']
    for d in delm:
        out_str = out_str.split(d)
        out_str = " ".join(out_str)
    return out_str


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)


@app.route('/prediction/<filename>', methods=['GET', 'POST'])
def prediction(filename):
    obj = DataObject
    obj.is_image_display = False
    obj.is_predicted = False
    if request.method == 'POST' and filename:
        val = app.config['UPLOADED_PHOTOS_DEST']+filename
        obj.image = filename
        if supported_image_type(val):
            filename = predict(val)
            if filename != "":
                obj.is_image_display = True
                obj.is_predicted = True
                obj.mask = filename
                return render_template('/predict.html', obj=obj)
            else:
                flash(f'Something went wrong with prediction. Try a different image')
            return render_template('/predict.html', obj=obj)
        else:
            flash(f'Uploaded image is not supported by our model. Try a different image')
    return render_template('/upload.html', obj=obj)


@app.route('/', methods=['GET', 'POST'])
def upload():
    obj = DataObject
    obj.is_image_display = False
    obj.image = ""
    if request.method == 'POST' and 'image' in request.files:
        image_file = request.files['image']
        if checkFileType(image_file.filename):
            filename = photos.save(image_file)
            obj.image = filename
            obj.is_image_display = True
            obj.is_predicted = False
            return render_template('/predict.html', obj=obj)
        else:
            if image_file.filename:
                msg = f"{image_file.filename} is not an image file"
            else:
                msg = "Please select an image file"
            flash(msg)
        return render_template('/upload.html', obj=obj)
    return render_template('/upload.html', obj=obj)



