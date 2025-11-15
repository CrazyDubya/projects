from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from .database import db
from .models import User

bp = Blueprint('auth', __name__, url_prefix='/auth')

@bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        error = None

        if not username:
            error = 'Username is required.'
        elif not email:
            error = 'Email is required.'
        elif not password:
            error = 'Password is required.'
        elif User.query.filter_by(username=username).first() is not None:
            error = f"User {username} is already registered."
        elif User.query.filter_by(email=email).first() is not None:
            error = f"Email {email} is already registered."

        if error is None:
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.')
            return redirect(url_for('auth.login'))

        flash(error)

    return render_template('auth/register.html')

@bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        error = None
        user = User.query.filter_by(email=email).first()

        if user is None or not user.check_password(password):
            error = 'Incorrect email or password.'

        if error is None:
            login_user(user)
            return redirect(url_for('index'))

        flash(error)

    return render_template('auth/login.html')

@bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))
