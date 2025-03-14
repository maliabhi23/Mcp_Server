# MCP Server

A Flask-based Middleware Communication Protocol (MCP) server that bridges API calls to GitHub. It fetches GitHub user information based on incoming POST requests.

## Features

- Fetch GitHub user details like username, full name, public repositories, followers, and profile URL.
- Error handling for invalid requests and GitHub user not found scenarios.

---

## Prerequisites

- Python 3.6 or above  
- GitHub API access (no authentication needed for basic usage but rate limits apply)

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mcp-server.git
cd mcp-server
