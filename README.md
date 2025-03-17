Установка зависимостей

```
curl -LsSf https://astral.sh/uv/install.sh | sh
pip install uv
uv venv
uv sync
uv pip install -e .
```

Пример запуска:

```
python .\src\pulsetrace\main.py --cfg config/config.yaml
```

TODOs:
+ 3 tf models
+ arima
SHAP local/global
Add ALIBI library XAI methods
pytorch model with custom class
png generate / html generate
