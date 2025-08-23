import os
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_login import LoginManager, current_user, login_required
from .database import db
from .models import User, Conversation, Vote, Feedback

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        SQLALCHEMY_DATABASE_URI='sqlite:///' + os.path.join(app.instance_path, 'shareconvo.db'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    with app.app_context():
        from . import models

    from . import cli
    cli.init_app(app)

    from . import auth
    app.register_blueprint(auth.bp)

    @app.route('/')
    def index():
        conversations = Conversation.query.all()
        return render_template('index.html', conversations=conversations)

    @app.route('/vote/<int:convo_id>/<string:vote_type>', methods=['POST'])
    @login_required
    def vote(convo_id, vote_type):
        conversation = Conversation.query.get_or_404(convo_id)

        # Check if the user has already voted on this conversation
        existing_vote = Vote.query.filter_by(user_id=current_user.id, conversation_id=convo_id).first()
        author = conversation.author

        if existing_vote:
            # If user is undoing their vote
            if existing_vote.vote_type == vote_type:
                if vote_type == 'like':
                    conversation.likes -= 1
                    author.add_points(-1)
                else:
                    conversation.dislikes -= 1
                db.session.delete(existing_vote)
            # If user is changing their vote
            else:
                if vote_type == 'like': # Changing dislike to like
                    conversation.dislikes -= 1
                    conversation.likes += 1
                    author.add_points(1)
                else: # Changing like to dislike
                    conversation.likes -= 1
                    conversation.dislikes += 1
                    author.add_points(-1)
                existing_vote.vote_type = vote_type
        else:
            # New vote
            new_vote = Vote(user_id=current_user.id, conversation_id=convo_id, vote_type=vote_type)
            db.session.add(new_vote)
            if vote_type == 'like':
                conversation.likes += 1
                author.add_points(1)
            else:
                conversation.dislikes += 1

        db.session.commit()

        return jsonify({'likes': conversation.likes, 'dislikes': conversation.dislikes})

    @app.route('/conversation/<int:convo_id>')
    def conversation_detail(convo_id):
        conversation = Conversation.query.get_or_404(convo_id)
        return render_template('conversation.html', conversation=conversation)

    @app.route('/feedback/<int:convo_id>', methods=['POST'])
    @login_required
    def submit_feedback(convo_id):
        data = request.get_json()

        # Validate data
        if not all(k in data for k in ['accuracy', 'rating', 'report']):
            return jsonify({'error': 'Missing data'}), 400

        # Get the conversation object first to avoid autoflush issues
        conversation = Conversation.query.get_or_404(convo_id)

        # Create feedback record
        feedback = Feedback(
            user_id=current_user.id,
            conversation_id=convo_id,
            description_accuracy=int(data['accuracy']),
            quality_rating=int(data['rating']),
            is_report=int(data['report'])
        )
        db.session.add(feedback)

        # We need to flush the new feedback to get an accurate count for the average
        db.session.flush()

        # Update conversation's quality score (simple running average)
        all_feedback = Feedback.query.filter_by(conversation_id=convo_id).all()
        total_ratings = sum(f.quality_rating for f in all_feedback)
        new_quality_score = total_ratings / len(all_feedback)
        conversation.quality_score = new_quality_score

        # Add points for giving feedback
        current_user.add_points(5)

        db.session.commit()

        return jsonify({'message': 'Feedback submitted successfully'}), 200

    @app.route('/submit', methods=['GET', 'POST'])
    @login_required
    def submit():
        if request.method == 'POST':
            title = request.form['title']
            blurb = request.form['blurb']
            full_text = request.form['full_text']
            certify = request.form.get('certify')

            if not all([title, blurb, full_text, certify]):
                # In a real app, we'd flash an error message
                return "Please fill out all fields and certify.", 400

            # Create conversation
            new_convo = Conversation(
                title=title,
                blurb=blurb,
                full_text=full_text,
                author=current_user
            )
            db.session.add(new_convo)

            # Add points for submission
            current_user.add_points(10)

            db.session.commit()

            return redirect(url_for('index'))

        return render_template('submit.html')

    return app
