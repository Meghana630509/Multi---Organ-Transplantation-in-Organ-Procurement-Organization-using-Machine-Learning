from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
from django.db import models
from django.core.exceptions import ValidationError
import re

# --- Custom Validators ---

def validate_name(value):
    if not re.fullmatch(r'[A-Za-z\s]{3,50}', value):
        raise ValidationError("Name must be 3-50 characters, letters and spaces only.")

def validate_mobile(value):
    if not re.fullmatch(r'\d{10}', value):
        raise ValidationError("Mobile number must be exactly 10 digits.")

def validate_password(value):
    if not re.fullmatch(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$', value):
        raise ValidationError(
            "Password must be at least 8 characters long and include uppercase, lowercase, digit, and special character."
        )
# --- Model ---

class donorRegisteredTable(models.Model):
    name = models.CharField(max_length=50, validators=[validate_name])
    email = models.EmailField(unique=True)
    loginid = models.CharField(max_length=30, unique=True)
    mobile = models.CharField(max_length=10, validators=[validate_mobile])
    password = models.CharField(max_length=128, validators=[validate_password])
    status = models.CharField(max_length=20, default='waiting')  # Default value added here

    def __str__(self):
        return self.name
