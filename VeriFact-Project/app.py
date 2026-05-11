from flask import Flask, request, jsonify, render_template
from main import fact_check

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/check', methods=['POST'])
def check():
    data = request.json
    claim = data.get('claim', '').strip()
    
    if not claim:
        return jsonify({"error": "Claim cannot be empty"}), 400
        
    try:
        result = fact_check(claim)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

