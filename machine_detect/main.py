from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, escape
import flask
import os
import sys
import matplotlib.pyplot as plt
import cv2

import image

app = Flask(__name__)

# @app.route("/")
# def hello():
#     name = request.args.get("name", "inhong")
#     return f"Hello, {escape(name)}!"

@app.route("/fileUpload", methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        return 'uploads'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

