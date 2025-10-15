from django.urls import path
from .views import *

app_name = 'App'

urlpatterns = [
    path('', home, name='home'),
    path('reg', reg, name='reg'),
    path('log', log, name='log'),
    path('logout', out, name='logout'),
    path('exam', exam, name='exam'),
    path('procotor', procotor, name='procotor'),
    path('exit_exam', exit_exam, name='exit_exam'),
    path('exam_code', exam_code_page, name='exam_code'),
    path('exam_result', exam_result_page, name='exam_result'),
    path('code_show', code_show, name='code_show'),
    path('api/tab_change', tab_change, name='tab_change'),
    path('question', question, name='question'),
    path('set_reminder/', set_reminder, name='set_reminder'),
     path('forgot_password', forgot_password, name='forgot_password'),

    

]

