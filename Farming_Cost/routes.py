# farming_cost/routes.py
from flask import Blueprint

farming_cost_bp = Blueprint('farming_cost', __name__)

@farming_cost_bp.route('/')
def index():
    return "Farming Cost Module"