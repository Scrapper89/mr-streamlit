mkdir /app/.streamlit
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > /app/.streamlit/config.toml
