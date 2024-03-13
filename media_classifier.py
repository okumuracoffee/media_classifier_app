import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np

## 分類したいクラス名を定義
classes = ["book", "CD"]
## 学習に用いた画像のサイズ
image_size = 50


UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

## Flaskクラスのインスタンス作成
app = Flask(__name__)

## アップロードされたアイルの拡張子のチェックをする
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./vgg16_model.h5')#学習済みモデルをロード


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            # load_img() 画像のロードとリサイズを同時に行う
            img = image.load_img(filepath, grayscale=False, target_size=(image_size,image_size))
            # image.img_to_array() Numpy配列に変換する
            img = image.img_to_array(img)
            data = np.array([img])
            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            print('result:', result)
            if result <= 0.5:
                predicted = 0
            else:
                predicted = 1
            #predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です" + "result:" + str(result) + "predicted:"+str(predicted)

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
