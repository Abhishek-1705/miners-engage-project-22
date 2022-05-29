from django.contrib import messages
from django.shortcuts import redirect, render
from django.contrib.auth.models import User, auth 
from django.http import HttpResponseRedirect

# Create your views here.


def login(request):
    if request.method == 'POST''GET':
        email=passwors=''
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = auth.authenticate(email=email,password=password)

        if user is not None:
            auth.login(request, user)
            return redirect('/movie')
        else:
            messages.info(request, "Invalid username and Password")
            return HttpResponseRedirect('login')
    else:        
        return render(request, 'login.html')


def signup(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        email = request.POST['email']
        if password1 == password2:
            if User.objects.filter(username=username).exists():
                messages.info(request, "Username already exists")
                return HttpResponseRedirect('signup')
            elif User.objects.filter(email=email).exists():
                messages.info(request, "email already exists")
                return HttpResponseRedirect('signup')
            else:
                user = User.objects.create_user(
                    username=username, first_name=first_name, last_name=last_name, password=password1, email=email)
                user.save()
                userlogin = auth.authenticate(email=email, password=password1)
                auth.login(request, user)
                return redirect('/movie')

        else:
            messages.info(request, "Password does not match ")
            return HttpResponseRedirect('signup')
        # return redirect('/')
    else:
        return render(request, 'signup.html')


def logout(request):
    auth.logout(request)
    return HttpResponseRedirect('/')
