from flask import Flask
from callers import usepath as up
import html
app = Flask(__name__)
 
@app.route('/hello/<path:name>')
def hello_name(name):
    #name=up.using_path(name)
    name=html.escape(name)
    return 'Hello %s!' % name

def main():
    print(hello_name('A'))

if __name__ == "__main__":
    main()


if __name__ == '__main__':
   app.run()