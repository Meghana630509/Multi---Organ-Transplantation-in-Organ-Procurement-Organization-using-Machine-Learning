from django.shortcuts import render
from django.views.decorators.cache import cache_control
# Create your views here.
from django.shortcuts import render
from django.contrib import messages

from Donors.models import donorRegisteredTable

@cache_control(no_cache=True,must_revalidate=True,no_store=True)
def adminHome(request):
    if not request.session.get('admin'):
        return render(request,'adminLoginForm.html')  
    return render(request, 'admin/adminHome.html')

def adminLoginCheck(request):
    if request.method=="POST":
        username=request.POST['adminUsername']
        adminPassword=request.POST['adminPassword']

        if adminPassword=="admin" and username=='admin':
            request.session['admin']=True
            return render(request,'admin/adminHome.html')
        else:
            messages.error(request,'Invalid details')
            return render(request,'adminLoginForm.html')
    else:
        return render(request,'adminLoginForm.html')
    

def donorList(request):
    if not request.session.get('admin'):
        return render(request,'adminLoginForm.html') 
    donors=donorRegisteredTable.objects.all()
    return render(request,'admin/donorList.html',{'donors':donors})
    if not request.session.get('admin'):
        return render(request,'adminLoginForm.html') 
    donors=donorRegisteredTable.objects.all()
    return render(request,'admin/donorList.html',{'donors':donors})


def log(request):
    request.session.flush()  # clears all session data
    return render(request,'index.html')


from django.shortcuts import redirect, get_object_or_404
from django.contrib import messages

def activate_donor(request):
    if not request.session.get('admin'):
        return render(request,'adminLoginForm.html') 

    id=request.GET['id']
    donor = get_object_or_404(donorRegisteredTable, id=id)
    donor.status = 'Active'
    donor.save()
    donors=donorRegisteredTable.objects.all()
    return render(request,'admin/donorList.html',{'donors':donors})  

def deactivate_donor(request):
    if not request.session.get('admin'):
        return render(request,'adminLoginForm.html') 
    
    id=request.GET['id']
    donor = get_object_or_404(donorRegisteredTable, id=id)
    donor.status = 'Inactive'
    donor.save()
    donors=donorRegisteredTable.objects.all()
    return render(request,'admin/donorList.html',{'donors':donors})

