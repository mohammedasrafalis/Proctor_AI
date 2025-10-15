from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import User

DEPT_CHOICES = [
    ('computer', 'Computer'),
    ('biology', 'Biology'),
]

TYPE_CHOICES = [
    ('student', 'Student'),
    ('teacher', 'Teacher'),
    ('admin', 'Admin')
]

class stu_register(AbstractUser):
    mobile = models.IntegerField(null=True)
    email = models.EmailField(unique=True)
    department = models.CharField(choices=DEPT_CHOICES, null=True, max_length=15)
    reg_type = models.CharField(choices=TYPE_CHOICES, null=True, max_length=15)

    def __str__(self):
        return self.username

class exam_code(models.Model):
    exam_number = models.IntegerField()

    def __str__(self):
        return str(self.exam_number)

class exam_result(models.Model):
    stu_id = models.ForeignKey(stu_register, on_delete=models.CASCADE)
    result = models.CharField(max_length=5, default='Pass')
    strike = models.IntegerField(default=0)
    Tab_strike = models.IntegerField(default=0)

    def __str__(self):
        return self.stu_id.username
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.conf import settings

class ExamQuestion(models.Model):
    teacher = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    exam_code = models.CharField(max_length=10)
    text = models.TextField()
    option1 = models.CharField(max_length=200)
    option2 = models.CharField(max_length=200)
    option3 = models.CharField(max_length=200)
    option4 = models.CharField(max_length=200)
    correct_answer = models.CharField(max_length=200)

    def __str__(self):
        return f"{self.exam_code}: {self.text}"

class questions(models.Model):
    depart = models.CharField(max_length=15)
    question1 = models.TextField()
    question2 = models.TextField()
    question3 = models.TextField()
    quest1_option1 = models.TextField()
    quest1_option2 = models.TextField()
    quest1_option3 = models.TextField()
    quest1_option4 = models.TextField()
    quest2_option1 = models.TextField()
    quest2_option2 = models.TextField()
    quest2_option3 = models.TextField()
    quest2_option4 = models.TextField()
    quest3_option1 = models.TextField()
    quest3_option2 = models.TextField()
    quest3_option3 = models.TextField()
    quest3_option4 = models.TextField()

    def __str__(self):
        return self.depart

from django.db import models
from django.conf import settings

class Profile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    whatsapp_number = models.CharField(max_length=20)

class Reminder(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    message = models.CharField(max_length=255)
    remind_at = models.DateTimeField()
    is_sent = models.BooleanField(default=False)
