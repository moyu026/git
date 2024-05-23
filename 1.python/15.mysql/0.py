import pymysql

# 连接MySQL数据库
conn = pymysql.connect(host='localhost', port=3306, user='root', password='', charset='utf8')
cursor = conn.cursor()

# 查看数据库
cursor.execute('SHOW DATABASES')
# 获取指令的结果
result = cursor.fetchall()
print(result) # (('information_schema',), ('db00',), ('mysql',), ('performance_schema',), ('sys',))

# 创建数据库
cursor.execute('create database db01 DEFAULT CHARSET utf8 COLLATE utf8_general_ci')
conn.commit()