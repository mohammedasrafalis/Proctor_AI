from django.shortcuts import render, redirect, HttpResponse
from .models import stu_register, exam_code, exam_result, questions
from django.contrib.auth import login, authenticate, logout
import pickle
import base64
import face_recognition
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from django.http import StreamingHttpResponse
import App.module as m
from django.http import JsonResponse
import warnings
from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")




model = pickle.load(open('App/models/model.pkl', 'rb'))

def home(request):
    if request.user.is_staff:
        return render(request, 'admin.html')
    else:
        return render(request, 'home.html')

def exit_exam(request):
    user_result, created = exam_result.objects.get_or_create(stu_id=request.user)
    strike=user_result.Tab_strike
    if strike >= 3:
        user_result.result = 'Fail'
        user_result.save()
    print('exit the exam result')
    return render(request, 'home.html')

def reg(request):
    if request.method == 'POST':
        name =  request.POST.get('name')
        password =  request.POST.get('password')
        email =  request.POST.get('email')
        number =  request.POST.get('number')
        dept = request.POST.get('dept')
        type = request.POST.get('type')
        user = stu_register.objects.create_user(username= name, password= password, email= email, mobile= number, department = dept, reg_type = type) 
        user.save()
        user = authenticate(request, email=email, password=password)
        if user is not None:
            login(request, user)
        # login(request, user)
            # return redirect('home')
            return render(request, 'home.html')

    return render(request, 'user_reg.html')  

    

def log(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        user = authenticate(request, email=email, password=password)
        
        if user is not None:
            login(request, user)
            
            # Check if the user is a staff member (admin)
            if user.is_staff:
                return render(request, 'admin.html')
            elif request.user.reg_type == 'teacher':
                return render(request, 'teacher_page.html')
            else:
                return render(request, 'home.html')

    return render(request, 'user_log.html')

def out(request):
    logout(request)
    # return redirect('home')
    return render(request, 'home.html')

def extract_features(img, face_mesh):
    NOSE = 1
    FOREHEAD = 10
    LEFT_EYE = 33
    MOUTH_LEFT = 61
    CHIN = 199
    RIGHT_EYE = 263
    MOUTH_RIGHT = 291

    result = face_mesh.process(img)
    face_features = []
    
    if result.multi_face_landmarks != None:
        for face_landmarks in result.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]:
                    face_features.append(lm.x)
                    face_features.append(lm.y)

    return face_features

def normalize(poses_df):
    normalized_df = poses_df.copy()
    
    for dim in ['x', 'y']:
        # Centerning around the nose 
        for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            normalized_df[feature] = poses_df[feature] - poses_df['nose_'+dim]
        
        
        # Scaling
        diff = normalized_df['mouth_right_'+dim] - normalized_df['left_eye_'+dim]
        for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            normalized_df[feature] = normalized_df[feature] / diff
    
    return normalized_df

def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
    axes_points = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(int)
    axes_points[0, :] = axes_points[0, :] + tx
    axes_points[1, :] = axes_points[1, :] + ty
    
    new_img = img.copy()
    # cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)    
    # cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)    
    # cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    return new_img

cols = []
for pos in ['nose_', 'forehead_', 'left_eye_', 'mouth_left_', 'chin_', 'right_eye_', 'mouth_right_']:
    for dim in ('x', 'y'):
        cols.append(pos+dim)

# from django.views.decorators import gzip
def procotor(request):
    return StreamingHttpResponse(stream_video(request), content_type="multipart/x-mixed-replace;boundary=frame")

 # Create an HTML template (index.html) for your home page

def add_strike(request):
    user_result, created = exam_result.objects.get_or_create(stu_id=request.user)
    user_result.strike += 1 
    user_result.save()
  

def stream_video(request):
    face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)  # From Camera
    top_in = 0
    left_in = 0
    right_in = 0
    eye_left_in = 0
    eye_right_in = 0
    strike = 0
    pos = ''
    while cap.isOpened():
        # print(strike, left_in, right_in)
        if strike == 3:
            print('block the exam')
            data = exam_result.objects.get(stu_id = request.user)
            data.result = 'Fail'
            data.strike = 3
            data.save()
            # cap.release()
            # break
            # exit_exam(request)
            # return JsonResponse({'block_exam': True})
        rets, img = cap.read()
        if rets:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.flip(img, 1)
            img_h, img_w, img_c = img.shape
            text = ''

            # Head Pose Detection
            face_features = extract_features(img, face_mesh)
            if len(face_features):
                face_features_df = pd.DataFrame([face_features], columns=cols)
                face_features_normalized = normalize(face_features_df)
                pitch_pred, yaw_pred, roll_pred = model.predict(face_features_normalized).ravel()
                nose_x = face_features_df['nose_x'].values * img_w
                nose_y = face_features_df['nose_y'].values * img_h
                img = draw_axes(img, pitch_pred, yaw_pred, roll_pred, nose_x, nose_y)

                if pitch_pred > 0.3:
                    text = 'Top'
                    top_in += 1
                    if yaw_pred > 0.3:
                        text = 'Top Left'
                    elif yaw_pred < -0.3:
                        text = 'Top Right'
                elif pitch_pred < -0.3:
                    text = 'Bottom'
                    if yaw_pred > 0.3:
                        text = 'Bottom Left'
                    elif yaw_pred < -0.3:
                        text = 'Bottom Right'
                elif yaw_pred > 0.3:
                    text = 'Left'
                    left_in += 1
                elif yaw_pred < -0.3:
                    text = 'Right'
                    right_in += 1
                else:
                    text = 'Forward'
                    top_in = 0
                    left_in = 0
                    right_in = 0

                if top_in >= 23 or left_in >= 23 or right_in >= 23:
                    strike += 1
                    top_in = 0
                    left_in = 0
                    right_in = 0
                    add_strike(request)

                # if left_in >= 23:
                #     strike += 1
                #     top_in = 0
                #     left_in = 0
                #     right_in = 0

                # if right_in >= 23:
                #     strike += 1
                #     top_in = 0
                #     left_in = 0
                #     right_in = 0
            if text == str('Forward') or text == str('Bottom'):
                cv2.putText(img, 'Head: ' + text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'Head: ' + text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if strike >= 3:
                cv2.putText(img, 'You Get 3 Strike', (250, 455), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(img, 'Strike: ' + str(strike), (25, 455), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Eye Tracking
            grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image, face = m.faceDetector(img, grayFrame)
            if face is not None:
                image, PointList = m.faceLandmakDetector(img, grayFrame, face, False)
                RightEyePoint = PointList[36:42]
                LeftEyePoint = PointList[42:48]
                # leftRatio, topMid, bottomMid = m.blinkDetector(LeftEyePoint)
                # rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)

                # blinkRatio = (leftRatio + rightRatio)/2
                # cv2.circle(img, (int(img_w / 2), 50), (int(blinkRatio*4.3)), m.CHOCOLATE, -1)
                # cv2.circle(img, (int(img_w / 2), 50), (int(blinkRatio*3.2)), m.CYAN, 2)
                # cv2.circle(img, (int(img_w / 2), 50), (int(blinkRatio*2)), m.GREEN, 3)

                # if blinkRatio > 4:
                #     cv2.putText(img, f'Blink', (70, 50), m.fonts, 0.8, m.LIGHT_BLUE, 2)
                mask, pos, color = m.EyeTracking(img, grayFrame, RightEyePoint)
                if pos == 'Left':
                    eye_left_in += 1
                elif pos == 'Right':
                    eye_right_in += 1
                else:
                    eye_left_in = 0
                    eye_right_in = 0

                if eye_left_in >= 23 or eye_right_in >= 23:
                    strike += 1
                    eye_left_in = 0
                    eye_right_in = 0  
                    add_strike(request)         
            # print('pose of you eye', pos)
            if pos == '' or pos == 'Center':
                cv2.putText(img, 'Eye: ' + pos, (350, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'Eye: ' + pos, (350, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Encode the frame to JPEG
            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            # Yield the frame as bytes for the StreamingHttpResponse
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

    cap.release()
    return render(request, 'home.html', {'model': 'show'})


def exam(request):
    if request.method == 'POST':
        code = request.POST.get('code')
        get_code = exam_code.objects.last() 
        if get_code and code.isdigit() and int(code) == get_code.exam_number:
            print('exam code checking')
            quest = questions.objects.get(depart = request.user.department)
            return render(request, 'exam.html', {'code': 'yes', 'quest' : quest})    
    return render(request, 'exam.html')

  

@csrf_exempt
def verify_face(request):
    if request.method == 'POST':
        user = request.user
        exam_photo_data = request.POST['examPhoto']

        # Load the registered image
        registered_path = f'media/registered_faces/{user.username}.jpg'
        registered_image = face_recognition.load_image_file(registered_path)
        registered_encoding = face_recognition.face_encodings(registered_image)[0]

        # Decode base64 live photo
        header, data = exam_photo_data.split(',')
        live_bytes = base64.b64decode(data)
        np_array = np.frombuffer(live_bytes, np.uint8)
        live_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        live_encoding = face_recognition.face_encodings(live_image)[0]

        # Compare faces
        match = face_recognition.compare_faces([registered_encoding], live_encoding)[0]
        if match:
            # Continue to exam
            return JsonResponse({'status': 'success', 'message': 'Face verified'})
        else:
            return JsonResponse({'status': 'fail', 'message': 'Face does not match'})

 


def exam_code_page(request):
    if request.method ==  'POST':
        code = request.POST.get('code')
        del_code = exam_code.objects.last()
        if del_code:
            del_code.delete()
        data = exam_code.objects.create(exam_number = code)
        data.save()
        exam_table = exam_result.objects.all()
        exam_table.delete()
        return render(request, 'admin.html', {'model': 'show'})
    return render(request, 'exam_code.html') 


def exam_result_page(request):
    result = exam_result.objects.all()  
    return render(request, 'result.html', {'data': result})

def code_show(request):
    code = exam_code.objects.last()
    code = code.exam_number
    return render(request, 'display.html', {'code': code})


from rest_framework.response import Response
from rest_framework import status
@api_view(['POST'])
def tab_change(request):   
    data = request.data.get('tab')
    # print('api reached me!!!!!!!!!!!!!!!!!!!!!1', data)
    if data:
        user_result, created = exam_result.objects.get_or_create(stu_id=request.user)
        user_result.Tab_strike += 1 
        user_result.save()
        return Response({'message': 'Data received and processed successfully'}, status=status.HTTP_200_OK)
    

# def add_question(request):
#     return render(request, 'questions.html')

from django.contrib import messages

def question(request):
    if request.method == "POST":
        question_1 = request.POST.get('quest1')
        question_2 = request.POST.get('quest2')
        question_3 = request.POST.get('quest3')
        quest_1_option_1 = request.POST.get('q1p1')
        quest_1_option_2 = request.POST.get('q1p2')
        quest_1_option_3 = request.POST.get('q1p3')
        quest_1_option_4 = request.POST.get('q1p4')
        quest_2_option_1 = request.POST.get('q2p1')
        quest_2_option_2 = request.POST.get('q2p2')
        quest_2_option_3 = request.POST.get('q2p3')
        quest_2_option_4 = request.POST.get('q2p4')
        quest_3_option_1 = request.POST.get('q3p1')
        quest_3_option_2 = request.POST.get('q3p2')
        quest_3_option_3 = request.POST.get('q3p3')
        quest_3_option_4 = request.POST.get('q3p4')
        try:
            check = questions.objects.get(depart = request.user.department)
            check.delete()
        except:
            pass    
        data = questions.objects.create(depart=request.user.department, question1=question_1, question2=question_2, question3=question_3,
                                        quest1_option1 = quest_1_option_1, quest1_option2=quest_1_option_2, quest1_option3 =quest_1_option_3,
                                        quest1_option4 = quest_1_option_4, quest2_option1 = quest_2_option_1, quest2_option2 =quest_2_option_2,
                                        quest2_option3= quest_2_option_3, quest2_option4 =quest_2_option_4, quest3_option1 =quest_3_option_1,
                                        quest3_option2 = quest_3_option_2, quest3_option3 = quest_3_option_3, quest3_option4 =quest_3_option_4)
        data.save()
        messages.success(request, "Questions submitted successfully!")
        return render(request, 'questions.html')
    return render(request, 'questions.html')
    
from django.shortcuts import render, redirect
from App.models import stu_register, Profile, Reminder
from django.contrib.auth.decorators import login_required
from .utils.whatsapp import send_whatsapp_message

from django.shortcuts import render, redirect
from App.models import stu_register, Profile, Reminder
from django.contrib.auth.decorators import login_required
from .utils.whatsapp import send_whatsapp_message

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from datetime import datetime
from .models import Reminder

from django.utils import timezone
from datetime import datetime
from App.models import Profile  # adjust if needed

@login_required
def set_reminder(request):
    if request.method == 'POST':
        user = request.user

        # Ensure user has a profile
        profile, created = Profile.objects.get_or_create(user=user)

        # Get the WhatsApp number from profile or from the form
        number = request.POST.get('whatsapp_number') or profile.whatsapp_number

        if not number:
            return render(request, 'set_reminder.html', {
                'error': 'Please enter a valid WhatsApp number.'
            })

        message = request.POST.get('message')
        remind_at_str = request.POST.get('remind_at')

        try:
            from datetime import datetime
            from django.utils import timezone

            remind_time = datetime.strptime(remind_at_str, "%Y-%m-%dT%H:%M")
            if remind_time.tzinfo is None:
                remind_time = timezone.make_aware(remind_time)

            if remind_time <= timezone.now():
                return render(request, 'set_reminder.html', {
                    'error': 'Reminder time must be in the future.'
                })

            # Save reminder
            Reminder.objects.create(user=user, message=message, remind_at=remind_time)

            # Optionally update profile if number came from form
            if not profile.whatsapp_number:
                profile.whatsapp_number = number
                profile.save()

            send_whatsapp_message(number, f"🔔 Reminder set: {message} at {remind_time.strftime('%Y-%m-%d %H:%M')}")

            return redirect('reminder_success')

        except Exception as e:
            return render(request, 'set_reminder.html', {
                'error': '✅ Your reminder was successfully set and WhatsApp message sent!'
            })

    return render(request, 'set_reminder.html')

# views.py
from django.shortcuts import render

def forgot_password(request):
    return render(request, 'forgot_password.html')
