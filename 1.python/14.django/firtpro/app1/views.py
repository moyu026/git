from django.shortcuts import render, HttpResponse, redirect

# Create your views here.

def index(request):
    return HttpResponse('第一次使用')

def user_list(request):
    # 去app目录下的templates目录下寻找user_list.html(根据app注册顺序，逐一去他们的templates目录中寻找)
    return render(request, 'user_list.html')

def tpl(request):
    name = '张'
    roles = ['admin', 'user']
    info = {'name': '张三', 'age': 18}
    data_list = [
        {'name': '张三', 'age': 18},
        {'name': '李四', 'age': 19}
    ]
    return render(request, 'tpl.html', {'name': name, 'roles': roles, 'info': info, 'data_list': data_list})

def req_resp(request):
    # request是一个对象，封装了用户发送过来的所有请求相关数据
    # 获取请求方式 GET/POST
    print(request.method)

    # 在url上传递值。
    print(request.GET)

    # 在请求体中提交数据
    print(request.POST)

    # 响应: HttpResponse('返回内容'), 字符串内容返回给请求者
    # return HttpResponse('返回内容')

    # 响应: 读取html的内容+渲染 ->字符串，返回给用户浏览器。
    # return render(request, 'tpl.html', {'title': '张三'})

    # 响应:让浏览器重定向到其他的页面
    return redirect('http://www.baidu.com')

def login(request):
    if request.method == 'GET':
        return render(request, 'login.html')
    else:
        # 如果是POST请求，获取用户提交的数据。
        # print(request.POST)
        username = request.POST.get('user')
        password = request.POST.get('pwd')
        if username == 'admin' and password == '123456':
            # return HttpResponse('登录成功')
            return redirect('http://www.baidu.com')
        else:
            # return HttpResponse('登录失败')
            return render(request, 'login.html', {'error_msg': '用户名或密码错误'})

def orm(request):
    # 测试orm操作表中的数据。
    from app1.models import UserInfo, Department

    # 新增数据
    Department.objects.create(title='开发部')
    Department.objects.create(title='测试部')
    Department.objects.create(title='运维部')
    UserInfo.objects.create(username='张三', password='123456', age=18, size=1, data=1)
    UserInfo.objects.create(username='李四', password='123456', age=19, size=2, data=2)
    UserInfo.objects.create(username='王五', password='123456', age=20, size=3, data=3)

    # 删除数据
    Department.objects.filter(id=1).delete()
    # Department.objects.all().delete()

    # # 获取数据
    # data_list = Department.objects.all()
    # print(data_list)
    # for obj in data_list:
    #     print(obj.id, obj.title)
    # # 获取第一条数据
    # obj = Department.objects.filter(id=1).first()
    # print(obj.id, obj.title)
    #
    # # 更新数据
    # UserInfo.objects.filter(id=1).update(age=18)
    # UserInfo.objects.all().update(age=19)

    return HttpResponse('成功')

def info_list(request):
    from app1.models import UserInfo, Department
    data_list = UserInfo.objects.all()
    return render(request, 'info_list.html', {'data_list': data_list})

def info_add(request):
    from app1.models import Department, UserInfo
    if request.method == 'GET':
        depart_list = Department.objects.all()
        return render(request, 'info_add.html', {'depart_list': depart_list})

    # 获取用户添加的数据
    username = request.POST.get('username')
    password = request.POST.get('password')
    age = request.POST.get('age')

    # 添加到数据库
    UserInfo.objects.create(username=username, password=password, age=age)

    # return HttpResponse('添加成功')
    return redirect('http://127.0.0.1:8000/info/list/')

def info_delete(request):
    from app1.models import UserInfo
    nid = request.GET.get('nid')
    UserInfo.objects.filter(id=nid).delete()
    # return HttpResponse('删除成功')
    return redirect('http://127.0.0.1:8000/info/list/')
