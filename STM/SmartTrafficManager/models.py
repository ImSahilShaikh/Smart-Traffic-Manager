from django.db import models

# Create your models here.


class RTO_Details(models.Model):
    license_plate = models.CharField(unique=True, primary_key=True, max_length=10)
    owner_name = models.CharField(max_length=20)
    owner_email = models.CharField(max_length=50)

    class Meta:
        db_table = "rto_details"


class violators(models.Model):
    license_plate = models.CharField(unique=True, primary_key=True,max_length=10)
    owner_name = models.CharField(max_length=20)
    owner_email = models.CharField(max_length=50)
    date_time = models.DateTimeField()
    email_status = models.BooleanField(default=False)

    class Meta:
        db_table = "violators"
