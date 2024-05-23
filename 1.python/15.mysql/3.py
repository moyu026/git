import pymysql

def register():
    print('用户注册')

    user = input('请输入用户名：')
    pwd = input('请输入密码：')

    # 连接指定数据库
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='', database='userdb', charset='utf8')
    cursor = conn.cursor()

    # 执行SQL语句
    sql = 'insert into users(name,password) values("{}","{}")'.format(user,pwd)

    cursor.execute(sql)
    conn.commit()

    # 关闭连接
    cursor.close()
    conn.close()
    print('注册成功')

def login():
    print('用户登录')

    user = input('请输入用户名：')
    pwd = input('请输入密码：')

    # 连接指定数据库
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='', database='userdb', charset='utf8')
    cursor = conn.cursor()

    # 执行SQL语句
    # sql = 'select * from users where name="{}" and password="{}"'.format(user,pwd)
    cursor.execute("select * from users where name=%(name)s and password=%(pwd)s",{'name':user,'pwd':pwd})

    # 获取查询结果
    result = cursor.fetchone()

    # 关闭连接
    cursor.close()
    conn.close()

    if result:
        print('登录成功')
    else:
        print('登录失败')

def run():
    choice = input('1.注册 2.登录 3.退出')
    if choice == '1':
        register()
    elif choice == '2':
        login()
    elif choice == '3':
        exit()
    else:
        print('输入错误')

if __name__ == '__main__':
    while True:
        run()