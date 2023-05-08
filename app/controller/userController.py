from app import db, app
from flask import jsonify, request
import time
from PIL import Image
import cv2
import torch
from numpy import ascontiguousarray, array, asarray, expand_dims, argmax
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, xywh2xyxy, clip_coords 
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from numpy import linalg
import numpy as np
import io
import base64
import dlib
from keras_facenet import FaceNet
import face_recognition_models
import face_recognition

predictor_5 = face_recognition_models.pose_predictor_five_point_model_location()
sp = dlib.shape_predictor(predictor_5)
MyFaceNet_new = FaceNet()

def facenet_signature(source,
           weights = 'yolov7-lite-t.pt',
           imgsz = 160,
           conf_thres = 0.25,
           iou_thres = 0.45,
           device = '',
           save_crop = False,
           classes = None,
           agnostic_nms = False,
           augment = False,
           project = 'runs/detect',
           name = 'exp',
           exist_ok = True,
           line_thickness = 3,
           hide_labels = False,
           hide_conf = False,
           kpt_label = 5):

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'   # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if isinstance(imgsz, (list,tuple)):
        assert len(imgsz) ==2; "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Read image
    #im0s = cv2.imread(source)
    #img0 = img # BGR
    im0s = array(Image.open(io.BytesIO(base64.b64decode(source))))
    im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
    assert im0s is not None, 'Image Not Found '

    # Padded resize
    img = letterbox(im0s, imgsz, stride=stride, auto=False)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = ascontiguousarray(img)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t_det_0 = time.time()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32, mengubah array menjadi float
    img /= 255.0  # mengubah angka aray dari 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0) #returns a new tensor with a dimension of size one inserted at the specified position dim.

    # Inference
    pred = model(img, augment=augment)[0] #Prediksi letak wajah pada gambar
    print(pred[...,4].max())
    
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms, kpt_label=kpt_label) 

    # Apply Classifier
    if classify: 
        pred = apply_classifier(pred, modelc, img, im0s)

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        s, im0 = '', im0s.copy()

        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
            scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            confi = 0
            # Write results
            for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                if conf > confi:
                    confi = conf
                
            for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):   
                # Add bbox to image
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                kpts = det[det_index, 6:]
                plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
            
                gain = 1.02
                pad = 10                        

                xyxy1 = torch.tensor(xyxy).view(-1, 4)
                b = xyxy2xywh(xyxy1)  # boxes
                b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
                xyxy1 = xywh2xyxy(b).long()
                clip_coords(xyxy1, im0.shape) #Memasukkan koordinat sebagai panjang dan tinggi gambar

                if conf == confi:
                    t_det_1 = time.time()              
                    #Convert koordinat tensor menjadi list
                    xyxy2 = xyxy1.squeeze().tolist()
                    x1,x2,x3,x4=xyxy2[0],xyxy2[1],xyxy2[2],xyxy2[3]

                    #Menentukan lokasi wajah dengan koordinat masukan
                    t_ext_0 = time.time()
                    face_location = dlib.rectangle(x1,x2,x3,x4)
                    faces = dlib.full_object_detections()

                    #Mendapatkan landmark dari setiap wajah
                    faces.append(sp(im0s, face_location))

                    #Menormalisasi bentuk wajah
                    image = dlib.get_face_chip(im0s, faces[0])

                    #Konversi BGR menjadi RGB
                    gb1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    #Konversi format gambar OpenCV menjadi PIL
                    gb1 = Image.fromarray(gb1)                       
                    gb1 = gb1.resize((160,160))
                    gb1 = asarray(gb1)
                    gb1 = expand_dims(gb1, axis=0) #Menambah dimensi

                    #Mengekstrak fitur wajah menjadi vektor dengen pre-trained model
                    signature = MyFaceNet_new.embeddings(gb1)
                    t_ext_1 = time.time()
                    return signature,xyxy1, (t_det_1-t_det_0), (t_ext_1-t_ext_0)
        else:
            return 'undetected',[],0,0

def ageitgey_signature(source,
           weights = 'yolov7-lite-t.pt',
           imgsz = 160,
           conf_thres = 0.25,
           iou_thres = 0.45,
           device = '',
           classes = None,
           agnostic_nms = False,
           augment = False,
           project = 'runs/detect',
           name = 'exp',
           exist_ok = True,
           line_thickness = 3,
           hide_labels = False,
           hide_conf = False,
           kpt_label = 5):
    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device) #Memilih menggunakan CPU atau GPU berdasarkan input
    half = device.type != 'cpu'   # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride

    if isinstance(imgsz, (list,tuple)): #cek apakah imgsz dalam bentuk tuple/list atau tidak
        assert len(imgsz) ==2; "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # mengubah model ke FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Read image
    # im0s = cv2.imread(source)
    im0s = array(Image.open(io.BytesIO(base64.b64decode(source))))
    im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
    assert im0s is not None, 'Image Not Found '

    # Padded resize
    img = letterbox(im0s, imgsz, stride=stride, auto=False)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = ascontiguousarray(img)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32, mengubah array menjadi float
    img /= 255.0  # mengubah angka aray dari 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0) #returns a new tensor with a dimension of size one inserted at the specified position dim.

    # Inference
    pred = model(img, augment=augment)[0] #Prediksi letak wajah pada gambar
    # print(pred[...,4].max())
    
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms, kpt_label=kpt_label) 

    # Apply Classifier
    if classify: 
        pred = apply_classifier(pred, modelc, img, im0s)

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        s, im0 = '', im0s.copy()

        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
            scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


            confi = 0
            # Write results
            for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                if conf > confi:
                    confi = conf

            for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                # Add bbox to image
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                kpts = det[det_index, 6:]
                plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
            
                gain = 1.02
                pad = 10                        

                xyxy1 = torch.tensor(xyxy).view(-1, 4)
                b = xyxy2xywh(xyxy1)  # boxes
                b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
                xyxy1 = xywh2xyxy(b).long()
                clip_coords(xyxy1, im0.shape) #Memasukkan koordinat sebagai panjang dan tinggi gambar
                
                if conf == confi:
                    t_det_1 = time.time()
                    #Memotong gambar dengan koordinat yang ditentukan sebelumnya
                    crop = im0[int(xyxy1[0, 1]):int(xyxy1[0, 3]), int(xyxy1[0, 0]):int(xyxy1[0, 2])] 
                    
                    #Konversi koordinat tensor menjadi tuple
                    xyxy2 = xyxy1.squeeze().tolist()
                    xyxy_list=[[xyxy2[1],xyxy2[2],xyxy2[3],xyxy2[0]]]
                    xyxy_tuple = [tuple(x) for x in xyxy_list]
                    t_ext_0 = time.time()
                    #Ekstrak wajah yang terdeteksi menjadi vektor
                    signature = face_recognition.face_encodings(im0s, xyxy_tuple, num_jitters=5, model = "large")
                    signature = expand_dims(signature, axis=0)
                    t_ext_1 = time.time()
                    det_time = t_det_1 - t0
                    ext_time = t_ext_1 - t_ext_0
                    return signature,xyxy1,det_time, ext_time
        else:
            return 'undetected',[],0,0


def save_to_db(Model, full_name, vector):
    try:
        new_user = Model(full_name=full_name, vector=vector)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'status': 'ok', 'message': 'Registrasi Berhasil'})
    except Exception as e:
        return jsonify({'status': 'failed', 'message': {str(e)}})
    
def verify_from_db(Model, min_dist, vector_input):
    db = Model.query.with_entities(Model.full_name,Model.vector)
    identity = 'unknown09123' 
    for rows in db:
        vector_db = rows.vector
        vector_db = np.fromstring(vector_db[2:-2], sep=' ') 
        dist = linalg.norm(vector_input - vector_db)
        if dist < min_dist:
            min_dist = dist
            identity = rows.full_name 

    return identity

def get_all_details(Model, key):
    query = Model.query.filter_by(full_name=key).first()
    return query