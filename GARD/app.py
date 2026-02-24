from flask import Flask, render_template, jsonify, request
from engine import GeodeticEngine

app = Flask(__name__)
engine = GeodeticEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/station_detail')
def station_detail():
    """Страница детального отчёта по станции. Данные передаются через sessionStorage."""
    return render_template('station_detail.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data          = request.json
        bbox          = data.get('bbox')
        target_count  = int(data.get('count', 15))
        target_epoch  = float(data.get('epoch', 2020.0))
        process_limit = int(data.get('process_limit', 0))

        result = engine.analyze_region(
            bbox,
            target_epoch=target_epoch,
            target_count=target_count,
            max_load=process_limit
        )

        if "error" in result:
            return jsonify({'status': 'error', 'message': result['error']}), 400

        return jsonify({'status': 'success', 'data': result})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
