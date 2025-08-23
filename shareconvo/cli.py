import click
from flask.cli import with_appcontext
from .database import db
from .models import User, Conversation

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    db.create_all()
    click.echo('Initialized the database.')

@click.command('seed-db')
@with_appcontext
def seed_db_command():
    """Seed the database with some sample data."""
    # Create a dummy user
    user = User.query.filter_by(email='test@example.com').first()
    if not user:
        user = User(username='testuser', email='test@example.com')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()

    # Create some dummy conversations
    convo1 = Conversation(
        title='Asking an AI to plan a mission to Mars with a $50 budget.',
        blurb='The results are... surprisingly detailed.',
        full_text='Full text of the conversation about the Mars mission.',
        author=user
    )
    convo2 = Conversation(
        title='The history of the rubber chicken.',
        blurb='A deep dive into the poultry-related prop.',
        full_text='Full text of the conversation about the rubber chicken.',
        author=user
    )
    db.session.add(convo1)
    db.session.add(convo2)
    db.session.commit()
    click.echo('Seeded the database.')


def init_app(app):
    app.cli.add_command(init_db_command)
    app.cli.add_command(seed_db_command)
