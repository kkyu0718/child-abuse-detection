import os
import cv2
import numpy as np 
import time
import datetime
import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from flask import flash

from model import *
import random
import shutil
import json
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision import transforms
from flask import Flask, url_for, redirect, render_template, request
from main import network_factory
from time_conversion import convert_datetime
from data.data_reader import DatasetReader
from data.data_splitter import DatasetSplit
from data.data_transformer import DatasetTransform
from data.transforms import SelectFrames, FrameDifference, Downsample, TileVideo, RandomCrop, Resize, RandomHorizontalFlip, Normalize, ToTensor

def graph_cctv_stats(violence,uncertain,end_time,duration=3, year=2021):
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches 
    plt.rcParams["figure.figsize"] = [10, 2]

    st_t = 0
    en_t = int(end_time)
    year = [year]

    for i in range(0,en_t,3):
        if i in violence:
            plt.barh(year, [duration], left=i, color="red")
        elif i in uncertain:
            plt.barh(year, [duration], left=i, color="yellow")
        else:
            plt.barh(year, [duration], left=i, color="green")

    plt.yticks([])
    plt.xticks([st_t]+violence+uncertain+[en_t])
    
    var_V = mpatches.Patch(color='red',label = 'violence')
    var_U = mpatches.Patch(color='yellow',label = 'uncertain')
    var_S = mpatches.Patch(color='green',label = 'safe')

    plt.legend(handles=[var_V,var_U,var_S], title="cctv statistics", loc=10, bbox_to_anchor=(0.5, 1.5),  ncol=3)
    plt.show()

def add_video_db(db, video_path, daycare_name, accuracy, status=0):
    daycare_center = DaycareCenter.query.filter_by(name=daycare_name).one()

    video = Video(
        detection_time = time.time() + 9 * 3600,
        name = os.path.basename(video_path),
        accuracy = round(accuracy,2),
        status = status,
        loc_id = daycare_center.loc_id,
        dc_id = daycare_center.id
        )

    db.session.add(video)
    db.session.commit()
    return video

def add_report_db(db, video_info):
    police_station = ['답십리지구대', '용신지구대', '청량리파출소', '제기파출소', '전농1파출소', '전농2파출소','장안1파출소', '장안2파출소', '이문지구대', '휘경파출소', '회기파출소']
    
    video = Video.query.filter_by(name=video_info.name).one()
    video.status = 1
    
    report_data = ReportList(
        time = time.time() + 9 * 3600, 
        police_name = np.random.choice(police_station, 1)[0],
        status = '출동 전',
        loc_id = video.loc_id,
        dc_id = video.dc_id,
        vid_id = video.id
    )
    db.session.add(report_data)
    db.session.commit()

    return report_data

def read_video(filename):
    frames = []
    if not os.path.isfile(filename):
        print('file not found')
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()  
        if not ret:
            break
        frames.append(frame)
    cap.release()
    video = np.stack(frames)
    return video

def load_model():
    model_path = '/content/gdrive/Shareddrives/2021청년인재_고려대과정_10조/Test Data/호준/호준model_best.E_bi_max_pool_ALL_fold_0_t2021-08-27 01:04:46.tar'
    seed = 250

    args = {
        'arch': 'E_bi_max_pool',
        'workers':1,
        'batch-size':1,
        'evalmodel':model_path,
        'kfold':0,
        'split':5,
        'frames':10,#5,#5,
        'lr':1e-6,
        'weight_decay':1e-1
    }

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    print("=> creating model '{}'".format(args['arch']))
    VP = network_factory(args['arch'])

    model = VP()
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args['lr'],
                                    weight_decay=args['weight_decay'])

    # optionally resume from a checkpoint
    if os.path.isfile(args['evalmodel']):
        print("=> loading checkpoint '{}'".format(args['evalmodel']))
        checkpoint = torch.load(args['evalmodel'])
        args['start_epoch'] = checkpoint['epoch']
        print('start_epoch : ', args['start_epoch'])
        best_prec = checkpoint['best_prec']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args['evalmodel'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args['evalmodel']))

    seed = 250
    args = {
        'arch': 'E_bi_max_pool',
        'workers':1,
        'batch-size':1,
        'evalmodel':model_path,
        'kfold':0,
        'split':5,
        'frames':10,#5,#5,
        'lr':1e-6,
        'weight_decay':1e-1
    }

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    print("=> creating model '{}'".format(args['arch']))
    VP = network_factory(args['arch'])

    model = VP()
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args['lr'],
                                    weight_decay=args['weight_decay'])

    # optionally resume from a checkpoint
    if os.path.isfile(args['evalmodel']):
        print("=> loading checkpoint '{}'".format(args['evalmodel']))
        checkpoint = torch.load(args['evalmodel'])
        args['start_epoch'] = checkpoint['epoch']
        print('start_epoch : ', args['start_epoch'])
        best_prec = checkpoint['best_prec']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args['evalmodel'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args['evalmodel']))

load_model()        
app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////content/gdrive/Shareddrives/2021청년인재_고려대과정_10조/Web/child_abuse_detection_database.db'
db = SQLAlchemy(app)
run_with_ngrok(app)   #starts ngrok when the app is run

saved_videos = []
daycare_center_name = '동대문어린이집'
starting_time = 0
end_time = 0
violence_threshold = 90
uncertain_threshold = 80

# 영상 저장을 위한 변수
save_video_path = '/content/gdrive/Shareddrives/2021청년인재_고려대과정_10조/Server/saved video/'
save_violence_path = '/content/gdrive/Shareddrives/2021청년인재_고려대과정_10조/Web/static/violence/'
save_uncertain_path = '/content/gdrive/Shareddrives/2021청년인재_고려대과정_10조/Web/static/uncertain/'

violence_files = []
uncertain_files = []
violence_time = []
uncertain_time = []

@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        global violence_files, violence_time, uncertain_files, uncertain_files, starting_time, end_time
        current_time = time.time()+9*60*60
        if starting_time == 0:
            starting_time = current_time
        end = time.time()

        file = request.files['file']
        frames = file.read()
        
        FILE_OUTPUT = daycare_center_name + '_' + str(time.strftime('%Y-%m-%d_%H_%M_%S %p', time.gmtime(current_time))) + '.mp4'
        saved_videos.append(FILE_OUTPUT)

        out_file = open(save_video_path + FILE_OUTPUT, "wb") 
        out_file.write(frames)
        out_file.close()

        video_path = save_video_path + FILE_OUTPUT
        
        video = read_video(video_path)
        val_dataset = DatasetReader(video)
        val_transformations = transforms.Compose([Resize(size=224), SelectFrames(num_frames=args['frames']), FrameDifference(dim=0), Normalize(), ToTensor()])

        val_dataset = DatasetTransform(val_dataset, val_transformations)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

        model.eval()

        for i, (input) in enumerate(val_loader):
            input_var = torch.autograd.Variable(input, volatile=True)
            input_var = input_var.cuda()

            # compute output
            output_dict = model(input_var)
            output = output_dict['classification']

            model_ret = output_dict['classification'].cpu().detach().numpy()[0][1] * 100
            if model_ret > violence_threshold:
                print('-----------------violence detected!!!!!!----------------')
                print(model_ret, '% violence detected')
                violence_files.append(FILE_OUTPUT)
                violence_time.append(current_time)
                shutil.copy(save_video_path+FILE_OUTPUT, save_violence_path+FILE_OUTPUT)
                save_name = save_violence_path+FILE_OUTPUT.split('.')[0] + '_' + str(round(model_ret,2)) +'.mp4'
                os.rename( save_violence_path+FILE_OUTPUT, save_name)

                # DB에 PUSH 하는부분
                video_info = add_video_db(db, save_name, daycare_center_name, model_ret, status=1)
                report_info = add_report_db(db,video_info)

            elif model_ret > uncertain_threshold:
                print('****** uncertainty detected ******')
                print(model_ret, '% violence detected')
                uncertain_files.append(FILE_OUTPUT)
                shutil.copy(save_video_path+FILE_OUTPUT, save_uncertain_path+FILE_OUTPUT)
                save_name = save_uncertain_path+FILE_OUTPUT.split('.')[0] + '_' + str(round(model_ret,2)) +'.mp4'
                os.rename( save_uncertain_path+FILE_OUTPUT,  save_name)

                # DB에 PUSH 하는부분
                video_info = make_video_db(db, save_name,daycare_center_name, model_ret)

            else:
                print('violence : ', round(model_ret,2),' %, - ',FILE_OUTPUT)
                os.rename(save_video_path+FILE_OUTPUT, save_video_path+FILE_OUTPUT.split('.')[0] + '_' + str(round(model_ret,2)) +'.mp4')
        print('model calulation time : ',round(time.time() - end,2), ' sec')
        end_time = current_time + 3

        if model_ret > violence_threshold:  
            return json.dumps(str(1))
        else:
            return json.dumps(str(0))

@app.route('/end_stream', methods=['GET'])
def end_stream():
    global violence_time, starting_time, end_time, uncertain_time
    violence = np.array(violence_time) - starting_time
    violence = violence.astype('uint32')
    uncertain = np.array(uncertain_time) - starting_time
    uncertain = uncertain.astype('uint32')
    endtime = end_time - starting_time
    graph_cctv_stats(violence,uncertain,endtime)

if __name__ == "__main__":
    app.run()

