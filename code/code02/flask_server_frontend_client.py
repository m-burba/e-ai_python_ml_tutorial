import requests

# Base URL of the API
base_url = 'http://127.0.0.1:5000'

# GET all items
response = requests.get(f'{base_url}/items')
print("GET /items:", response.json())

# GET a specific item (e.g., id = 1)
response = requests.get(f'{base_url}/items/1')
print("GET /items/1:", response.json())

# POST a new item
new_item = {'name': 'New Item'}
response = requests.post(f'{base_url}/items', json=new_item)
print("POST /items:", response.json())

# PUT to update an item (e.g., id = 1)
updated_item = {'name': 'Updated Item 1'}
response = requests.put(f'{base_url}/items/1', json=updated_item)
print("PUT /items/1:", response.json())

# DELETE an item (e.g., id = 1)
response = requests.delete(f'{base_url}/items/1')
print("DELETE /items/1:", response.json())

# Check items after deletion
response = requests.get(f'{base_url}/items')
print("GET /items after deletion:", response.json())

# -------------------------
# UPLOAD a file for an item (e.g., for item with id = 2)
upload_url = f'{base_url}/items/2/upload'
# Ensure you have a file named 'example.txt' in your current directory
with open('example.txt', 'rb') as f:
    files = {'file': f}
    response = requests.post(upload_url, files=files)
    print("POST /items/2/upload:", response.json())

# -------------------------
# DOWNLOAD the file associated with an item (e.g., for item with id = 2)
download_url = f'{base_url}/items/2/download'
response = requests.get(download_url, stream=True)
if response.status_code == 200:
    # Save the downloaded file locally
    with open('downloaded_example.txt', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("File downloaded successfully and saved as downloaded_example.txt")
else:
    print("Failed to download file, status code:", response.status_code)
