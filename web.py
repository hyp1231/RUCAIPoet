import json
from flask import Flask, render_template, request
from multi_wd import api


class CustomFlask(Flask):
    jinja_options = Flask.jinja_options.copy()
    jinja_options.update(dict(
        block_start_string='(%',
        block_end_string='%)',
        variable_start_string='((',
        variable_end_string='))',
        comment_start_string='(#',
        comment_end_string='#)',
    ))


app = CustomFlask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/input_poet', methods=['POST'])
def input_poet():
    input_poet = request.form['input']
    print(input_poet, flush=True)
    output_poet = api(input_poet, 10)
    return json.dumps({'output': output_poet})

app.run(host='0.0.0.0', port=8848, debug=True)
