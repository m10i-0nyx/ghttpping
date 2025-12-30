# ghttpping
A simple HTTP ping tool to check the availability and response time of web servers.

# Installation
```bash
git clone https://github.com/m10i-0nyx/ghttpping.git
cd ghttpping
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -p 3.14 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

# Usage
```bash
python ghttpping.py [options] <URL> 
```
