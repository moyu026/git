from flask import Flask,render_template
app = Flask(__name__)

# 创建了网址/show/info 和函数 index()的对应关系
# 在浏览器上访问/show/info ，网站自动执行函数index()
@app.route('/show/info')
def idex():
    return render_template('1.html')

if __name__ == '__main__':
    app.run()