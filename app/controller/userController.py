from app.dbmodel.user import User
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
from keras.models import load_model
from numpy import linalg
import numpy as np
import io
import base64


MyFaceNet = load_model('facenet_keras.h5')

def create_signature(source,
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
           kpt_label = 5,
           create_signature = False,
           recognize = True):

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

    # Set Dataloader
    vid_path, vid_writer = None, None

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
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]
    print(pred[...,4].max())
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms, kpt_label=kpt_label)
    t2 = time_synchronized()

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


            # Write results
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
                clip_coords(xyxy1, im0.shape)
                crop = im0[int(xyxy1[0, 1]):int(xyxy1[0, 3]), int(xyxy1[0, 0]):int(xyxy1[0, 2])] 

                gb1 = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                gb1 = Image.fromarray(gb1)                       
                gb1 = gb1.resize((160,160))
                gb1 = asarray(gb1)
                
                gb1 = gb1.astype('float32')
                mean, std = gb1.mean(), gb1.std()
                gb1 = (gb1 - mean) / std

                gb1 = expand_dims(gb1, axis=0)
                signature = MyFaceNet.predict(gb1)

                xyxy_res=xyxy1.numpy()
                return signature,xyxy1
        else:
            return 'gagal',[]


def save_to_db(full_name, identity_number, vector, gender, birth, address, email):
    try:
        new_user = User(full_name=full_name, identity_number=identity_number, vector=vector, gender=gender, birth=birth, address=address, email=email)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'status': 'ok', 'message': 'Registrasi Berhasil'})
    except Exception as e:
        return jsonify({'status': 'failed', 'message': {str(e)}})
    
def verify_from_db(min_dist, vector_input):
    db = User.query.with_entities(User.full_name,User.vector)
    identity = 'unknown' 
    for rows in db:
        vector_db = rows.vector
        vector_db = np.fromstring(vector_db[2:-2], sep=' ') 
        dist = linalg.norm(vector_input - vector_db)
        print(dist)
        if dist < min_dist:
            min_dist = dist
            identity = rows.full_name 
            
    return identity

def get_all_details(key):
    query = User.query.filter_by(full_name=key).first()
    return query