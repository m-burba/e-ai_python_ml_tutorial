from flask import Flask, jsonify, request, abort, send_from_directory
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# In-memory "database" of items; each item can optionally have a 'file' key.
items = [
    {"id": 1, "name": "Item 1"},
    {"id": 2, "name": "Item 2"},
]

# GET /items - Retrieve all items
@app.route('/items', methods=['GET'])
def get_items():
    return jsonify(items)

# GET /items/<id> - Retrieve a specific item by its id
@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = next((item for item in items if item['id'] == item_id), None)
    if item is None:
        abort(404)  # Not Found
    return jsonify(item)

# POST /items - Create a new item (auto-generates id)
@app.route('/items', methods=['POST'])
def create_item():
    if not request.json or 'name' not in request.json:
        abort(400)  # Bad Request
    new_item = {
        "id": items[-1]["id"] + 1 if items else 1,
        "name": request.json['name']
    }
    items.append(new_item)
    return jsonify(new_item), 201  # 201 Created

# PUT /items/<id> - Update an existing item or create a new one with a chosen id
@app.route('/items/<int:item_id>', methods=['PUT'])
def update_or_create_item(item_id):
    if not request.json or 'name' not in request.json:
        abort(400)  # Bad Request
    item = next((item for item in items if item['id'] == item_id), None)
    if item is None:
        # Create a new item with the specified id
        new_item = {"id": item_id, "name": request.json['name']}
        items.append(new_item)
        return jsonify(new_item), 201  # Created new item
    else:
        # Update the existing item
        item['name'] = request.json.get('name', item['name'])
        return jsonify(item)

# DELETE /items/<id> - Delete an item
@app.route('/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    global items
    items = [item for item in items if item['id'] != item_id]
    return jsonify({'result': True})

# POST /items/<id>/upload - Upload a file associated with an item
@app.route('/items/<int:item_id>/upload', methods=['POST'])
def upload_file(item_id):
    item = next((item for item in items if item['id'] == item_id), None)
    if item is None:
        abort(404)
    if 'file' not in request.files:
        abort(400, description="No file part in the request")
    file = request.files['file']
    if file.filename == '':
        abort(400, description="No selected file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Prepend the item id to avoid collisions
        saved_filename = f"{item_id}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, saved_filename)
        file.save(file_path)
        # Save file info in the item
        item['file'] = file_path
        return jsonify({'result': 'File uploaded', 'file_path': file_path}), 201
    else:
        abort(400, description="File type not allowed")

# GET /items/<id>/download - Download the file associated with an item
@app.route('/items/<int:item_id>/download', methods=['GET'])
def download_file(item_id):
    item = next((item for item in items if item['id'] == item_id), None)
    if item is None:
        abort(404)
    if 'file' not in item:
        abort(404, description="No file available for this item")
    file_path = item['file']
    directory, filename = os.path.split(file_path)
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
