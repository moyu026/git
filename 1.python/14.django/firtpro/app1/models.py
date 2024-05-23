from django.db import models

class UserInfo(models.Model):
    username = models.CharField(max_length=32)
    password = models.CharField(max_length=64)
    age = models.IntegerField()

    size = models.IntegerField(default=2)
    data = models.IntegerField(null=True, blank=True)

class Department(models.Model):
    title = models.CharField(max_length=32)

# 新增数据
# UserInfo.objects.create(username='alex', password='123', age=18)
