from django.shortcuts import render

def deapart_list(request):
    # 部门列表
    return render(request, 'deapart_list.html')
