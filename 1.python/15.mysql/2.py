import pymysql

# 连接数据库，自动执行use db02; --进入数据库
conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='', db='db02')
cursor = conn.cursor()

# 新增
cursor.execute("insert into student values('1','xiaozhang','22','90','xxxc',NOW())")
conn.commit()

# 删除
cursor.execute("delete from student where id='0'")
conn.commit()

# 修改
cursor.execute("update student set name='xiaoli' where id='1'")
conn.commit()

# 关闭连接
cursor.close()
conn.close()