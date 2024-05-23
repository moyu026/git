import pymysql

# 创建连接
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='',
    charset='utf8'
)
cursor = conn.cursor()

# 创建数据库
cursor.execute('create database db02 default charset utf8 collate  utf8_general_ci')
conn.commit()

# 进入数据库，查看数据表
cursor.execute('use db02')
cursor.execute('show tables')
print(cursor.fetchall())

# 进入数据库创建数据表
cursor.execute('use db02')
sql = """
create table student(
    id int primary key auto_increment,
    name varchar(20),
    age int,
    score float,
    content text,
    ctime datetime
) default charset=utf8;
"""
cursor.execute(sql)
conn.commit()
cursor.execute('show tables')

# 关闭连接
cursor.close()
conn.close()