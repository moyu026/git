from django.db import models


class Department(models.Model):
    # 部门表
    title = models.CharField(verbose_name='标题', max_length=32)


class UserInfo(models.Model):
    # 员工表
    name = models.CharField(verbose_name='姓名', max_length=32)
    password = models.CharField(verbose_name='密码', max_length=64)
    age = models.IntegerField(verbose_name='年龄')
    account = models.DecimalField(verbose_name='账户余额', max_digits=10, decimal_places=2, default=0)
    create_time = models.DateTimeField(verbose_name='创建时间')

    # 无约束""
    #depart_id = models.BigIntegerField(verbose_name='部门id')

    # 有约束
    # -to 与表关联
    # -to_field 与表中的那一列关联
    # -on_delete 级联删除
    depart = models.ForeignKey(to='Department', to_field='id', on_delete=models.CASCADE)
    # depart = models.ForeignKey(to='Department', to_field='id',null=True,blank=True, on_delete=models.SET_NULL) # 允许为空

    gender_choices = (
        (1, '男'),
        (2, '女')
    )
    gender = models.SmallIntegerField(verbose_name='性别', choices=gender_choices)