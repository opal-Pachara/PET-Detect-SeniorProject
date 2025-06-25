from flask import Blueprint, request, render_template_string, redirect, url_for, session
from .db import mongo

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = mongo.db.users.find_one({'username': username})
        if user:
            return 'Username already exists!'
        mongo.db.users.insert_one({'username': username, 'password': password})
        return redirect(url_for('auth.login'))
    return render_template_string('''
        <form method="post">
            Username: <input name="username"><br>
            Password: <input name="password" type="password"><br>
            <input type="submit" value="Register">
        </form>
    ''')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = mongo.db.users.find_one({'username': username, 'password': password})
        if user:
            session['user_id'] = str(user['_id'])
            return 'Login Success!'
        else:
            return 'Login Failed!'
    return render_template_string('''
        <form method="post">
            Username: <input name="username"><br>
            Password: <input name="password" type="password"><br>
            <input type="submit" value="Login">
        </form>
    ''') 