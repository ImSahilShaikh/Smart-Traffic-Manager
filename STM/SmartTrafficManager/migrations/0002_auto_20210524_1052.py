# Generated by Django 3.1a1 on 2021-05-24 05:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SmartTrafficManager', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='rto_details',
            name='owner_email',
            field=models.CharField(max_length=50),
        ),
        migrations.AlterField(
            model_name='violators',
            name='owner_email',
            field=models.CharField(max_length=50),
        ),
    ]
