from app import app
from flask import Flask, request, jsonify
from numpy import linalg, array
from PIL import Image
from datetime import datetime
import numpy as np
import io
import cv2
import base64

from app.controller import userController

@app.route('/')
def index():
    return jsonify({'status':'ok', 'message':'API is ready to use'})

@app.route('/verification', methods=['POST'])
def verification():   
    try:
        # Menerima data dari frontend
        input = request.get_json(force=True)
        img_input = input['img'].replace('data:image/png;base64,','')    
        
        # Membuat vektor dan koordintar bounding box
        vector_input, xyxy = userController.create_signature(source=img_input)
        if vector_input == 'gagal':
            return jsonify({'status' : 'failed', 'message': 'wajah tidak terdeteksi'})

        # Menggambar bounding box
        im0s = array(Image.open(io.BytesIO(base64.b64decode(img_input))))
        im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
        c1, c2 = (int(xyxy[0][0]), int(xyxy[0][1])), (int(xyxy[0][2]), int(xyxy[0][3]))
        cv2.rectangle(im0s, c1, c2, (255,0,0), lineType=cv2.LINE_AA)  

        # Konversi gambar ke byte string
        _, im_arr = cv2.imencode('.jpg', im0s) 
        im_bytes = im_arr.tobytes()
        im_output_b64 = base64.b64encode(im_bytes)
        im_output_b64_str = im_output_b64.decode("utf-8")     
        
        # Mencari identitas yang sesuai
        min_dist=10
        identity= userController.verify_from_db(min_dist, vector_input)

        if input['nama']==identity:
            details = userController.get_all_details(identity)
            response = {
                'status' : 'ok',
                'nama' : details.full_name,
                'id_num' : details.identity_number,
                'gender' : details.gender,
                'birth' : details.birth,
                'address' : details.address,
                'email' : details.email,
                'img' : im_output_b64_str
            }
            return jsonify(response)
        else:
            return jsonify({'status' : 'failed', 'message':'Identitas tidak sesuai'})
    
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/registration', methods=['POST'])
def registration():
    try:
        # Mennerima data dari frontend
        req_data = request.get_json(force=True)
        img_input = req_data['image'].replace('data:image/png;base64,','')

        # Membuat vektor identitas
        vector, xyxy = userController.create_signature(source=img_input)
        if vector == 'gagal':
            return jsonify({'status': 'failed', 'message': 'Wajah tidak terdeteksi'})
        
        # Menggabungkan tempat dan tanggal lahir
        date_obj = datetime.strptime(req_data['birthdate'], '%Y-%m-%d')
        birthdate = date_obj.strftime('%d %B %Y')
        birthplace = req_data['birthplace'].capitalize()

        # Menyimpan data dengan format yang sesuai
        birth = f'{birthplace}, {birthdate}'
        full_name = req_data['name']
        identity_number = req_data['id_num']
        vector = str(vector)
        gender = req_data['gender']
        address = req_data['address']
        email = req_data['email']

        # Menympan data ke database
        return userController.save_to_db(full_name, identity_number, vector, gender, birth, address, email)

    except Exception as e:
        return jsonify({"status":"failed", "message": str(e)})