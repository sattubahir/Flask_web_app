from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('my_model.keras')
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Form submitted, process the image and show result on the same page
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            img = Image.open(io.BytesIO(file.read()))
            img = img.resize((150, 150))
            img = image.img_to_array(img)
            img = img.reshape(1, 150, 150, 3)
            result = model.predict(img)
            predicted_class = labels[np.argmax(result)]
            return render_template_string("""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Cerebral Neoplasm Detection</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            background-color: #f4f4f4;
                            margin: 0;
                            padding: 0;
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            height: 100vh;
                        }

                        h1 {
                            text-align: center;
                            color: #333;
                            margin-bottom: 20px;
                        }

                        form {
                            text-align: center;
                        }

                        input[type="file"] {
                            padding: 10px;
                            margin-bottom: 10px;
                            border: 1px solid #ddd;
                            border-radius: 5px;
                            background-color: #fff;
                            cursor: pointer;
                        }

                        input[type="submit"] {
                            padding: 10px 20px;
                            background-color: #4caf50;
                            color: #fff;
                            border: none;
                            border-radius: 5px;
                            cursor: pointer;
                        }

                        input[type="submit"]:hover {
                            background-color: #45a049;
                        }

                        p {
                            margin-top: 20px;
                            font-size: 18px;
                            color: #333;
                        }
                    </style>
                </head>
                <body>
                    <h1>Cerebral Neoplasm Detection</h1>
                    <form action="/" method="post" enctype="multipart/form-data">
                        <input type="file" name="file" accept=".jpg, .jpeg, .png">
                        <br>
                        <input type="submit" value="Predict">
                    </form>
                    <p>Prediction: {{ predicted_class }}</p>
                </body>
                </html>
            """, predicted_class=predicted_class)

    # Render the initial page
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Cerebral Neoplasm Detection</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    height: 100vh;
                }

                h1 {
                    text-align: center;
                    color: #333;
                    margin-bottom: 20px;
                }

                form {
                    text-align: center;
                }

                input[type="file"] {
                    padding: 10px;
                    margin-bottom: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: #fff;
                    cursor: pointer;
                }

                input[type="submit"] {
                    padding: 10px 20px;
                    background-color: #4caf50;
                    color: #fff;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }

                input[type="submit"]:hover {
                    background-color: #45a049;
                }
            </style>
        </head>
        <body>
            <h1>Cerebral Neoplasm Detection</h1>
            <form action="/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".jpg, .jpeg, .png">
                <br>
                <input type="submit" value="Predict">
            </form>
        </body>
        </html>
    """)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
