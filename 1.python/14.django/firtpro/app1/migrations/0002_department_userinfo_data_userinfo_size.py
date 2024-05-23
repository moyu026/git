# Generated by Django 4.2.11 on 2024-05-07 02:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app1', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Department',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=32)),
            ],
        ),
        migrations.AddField(
            model_name='userinfo',
            name='data',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='userinfo',
            name='size',
            field=models.IntegerField(default=2),
        ),
    ]
