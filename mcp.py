from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
GITHUB_API_URL = "https://api.github.com/users/"

@app.route('/mcp/github', methods=['POST'])
def github_mcp():
    data = request.json
    action = data.get("action")
    username = data.get("parameters", {}).get("username", "")
    request_id = data.get("id", "unknown")
    
    if action == "get_user_info" and username:
        response = requests.get(GITHUB_API_URL + username)
        if response.status_code == 200:
            user_data = response.json()
            return jsonify({
                "id": request_id,
                "status": "success",
                "content": {
                    "username": user_data.get("login"),
                    "name": user_data.get("name"),
                    "public_repos": user_data.get("public_repos"),
                    "followers": user_data.get("followers"),
                    "profile_url": user_data.get("html_url")
                }
            })
        else:
            return jsonify({"id": request_id, "status": "error", "message": "GitHub user not found"}), 404
    
    return jsonify({"id": request_id, "status": "error", "message": "Invalid request"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)  
