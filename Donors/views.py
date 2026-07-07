from django.shortcuts import render
import os


from Donors.utility.classification import classification_View,predict_with_best_model


from .models import donorRegisteredTable
from django.core.exceptions import ValidationError
from django.contrib import messages


def donorRegisterCheck(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        username = request.POST.get("loginId")
        mobile = request.POST.get("mobile")
        password = request.POST.get("password")
        

        # Create an instance of the model
        donor = donorRegisteredTable(
            name=name,
            email=email,
            loginid=username,
            mobile=mobile,
            password=password,
            
        )

        try:
            # Validate using model field validators
            donor.full_clean()
            
            # Save to DB
            donor.save()
            messages.success(request,'registration Successfully done,please wait for admin APPROVAL')
            return render(request, "donorRegisterForm.html")


        except ValidationError as ve:
            # Get a list of error messages to display
            error_messages = []
            for field, errors in ve.message_dict.items():
                for error in errors:
                    error_messages.append(f"{field.capitalize()}: {error}")
            return render(request, "donorRegisterForm.html", {"messages": error_messages})

        except Exception as e:
            # Handle other exceptions (like unique constraint fails)
            return render(request, "donorRegisterForm.html", {"messages": [str(e)]})

    return render(request, "donorRegisterForm.html")


def donorLoginCheck(request):
    if request.method=='POST':
        username=request.POST['donorUsername']
        password=request.POST['donorPassword']

        try:
            donor=donorRegisteredTable.objects.get(loginid=username,password=password)

            if donor.status=='Active':
                request.session['id']=donor.id
                request.session['name']=donor.name
                request.session['email']=donor.email
                
                return render(request,'donors/donorHome.html')
            else:
                messages.error(request,'Status not activated please wait for admin approval')
                return render(request,'donorLoginForm.html')
        except:
            messages.error(request,'Invalid details please enter details carefully or Please Register')
            return render(request,'donorLoginForm.html')
    return render(request,'donorLoginForm.html')


def donorHome(request):
    if not request.session.get('id'):
        return render(request,'donorLoginForm.html')
    return render(request,'donors/donorHome.html')


def training(request):
    if not request.session.get('id'):
        return render(request,'donorLoginForm.html')

    metrics = classification_View()
   
    return render(request, 'donors/training.html',metrics)


from django.shortcuts import render
import numpy as np
import joblib
import os

from tensorflow.keras.models import load_model





def prediction(request):
    if not request.session.get('id'):
        return render(request, 'donorLoginForm.html')

    if request.method == 'POST':
        try:
            # Collect the form values
            fields = [
                'Age', 'Referral_Year', 'Procured_Year',
                'brain_death_True', 'Mechanism_of_Death_ICH/Stroke',
                'ABO_BloodType_A1', 'ABO_BloodType_O',
                'approached_True', 'Eye_Referral_True',
                'referral_to_approach_within_24h'
            ]
            input_data = []
            for field in fields:
                val = request.POST.get(field)
                input_data.append(float(val))

            # Load scaler
            scaler = joblib.load('models/scaler.pkl')
            

            
            best_model_name = 'LR'  
            if best_model_name == 'ANN':
                model = load_model(f'models/{best_model_name}_model.h5')
            else:
                model = joblib.load(f'models/{best_model_name}_model.pkl')

            # Make prediction
            result = predict_with_best_model(best_model_name, model, scaler, input_data)

            return render(request, 'donors/prediction.html', {
                "input": dict(zip(fields, input_data)),
                "pt": result
            })

        except Exception as e:
            print("Error during prediction:", e)
            return render(request, 'donors/prediction.html', {
                'error': "An error occurred during prediction."
            })

    return render(request, 'donors/prediction.html')




    
