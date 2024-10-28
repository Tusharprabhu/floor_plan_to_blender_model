from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/generate-3d', methods=['GET'])
def generate_3d():
    try:
        # Simulate success response
        return jsonify(status="success", output="Blender project created successfully!")
    except Exception as e:
        return jsonify(status="error", output=str(e))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)