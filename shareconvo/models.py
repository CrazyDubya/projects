from .database import db
from datetime import datetime
from flask_login import UserMixin
import bcrypt

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    points = db.Column(db.Integer, nullable=False, default=0)
    level = db.Column(db.Integer, nullable=False, default=1)
    conversations = db.relationship('Conversation', backref='author', lazy=True)
    votes = db.relationship('Vote', backref='voter', lazy=True)
    feedback = db.relationship('Feedback', backref='reviewer', lazy=True)

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

    def add_points(self, amount):
        self.points += amount
        # Simple level calculation: 1 level per 100 points
        self.level = (self.points // 100) + 1

class Conversation(db.Model):
    __tablename__ = 'conversations'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(120), nullable=False)
    blurb = db.Column(db.Text, nullable=False)
    full_text = db.Column(db.Text, nullable=False)
    created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    likes = db.Column(db.Integer, nullable=False, default=0)
    dislikes = db.Column(db.Integer, nullable=False, default=0)
    quality_score = db.Column(db.Float, nullable=False, default=0.0)
    votes = db.relationship('Vote', backref='conversation', lazy=True)
    feedback = db.relationship('Feedback', backref='conversation_feedback', lazy=True)

class Vote(db.Model):
    __tablename__ = 'votes'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=False)
    vote_type = db.Column(db.String(10), nullable=False) # 'like' or 'dislike'
    created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class Feedback(db.Model):
    __tablename__ = 'feedback'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=False)
    description_accuracy = db.Column(db.Integer, nullable=False) # 1 for accurate, 0 for inaccurate
    quality_rating = db.Column(db.Integer, nullable=False) # 1-5
    is_report = db.Column(db.Integer, nullable=False, default=0)
    created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
