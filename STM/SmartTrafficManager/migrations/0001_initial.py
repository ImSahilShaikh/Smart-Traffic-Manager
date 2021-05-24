# Generated by Django 3.1a1 on 2021-05-24 04:53

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='RTO_Details',
            fields=[
                ('license_plate', models.CharField(max_length=10, primary_key=True, serialize=False, unique=True)),
                ('owner_name', models.CharField(max_length=20)),
                ('owner_email', models.CharField(max_length=20)),
            ],
            options={
                'db_table': 'rto_details',
            },
        ),
        migrations.CreateModel(
            name='violators',
            fields=[
                ('license_plate', models.CharField(max_length=10, primary_key=True, serialize=False, unique=True)),
                ('owner_name', models.CharField(max_length=20)),
                ('owner_email', models.CharField(max_length=20)),
                ('date_time', models.DateTimeField()),
                ('email_status', models.BooleanField(default=False)),
            ],
            options={
                'db_table': 'violators',
            },
        ),
    ]