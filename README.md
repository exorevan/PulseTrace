Установка зависимостей

```
curl -LsSf https://astral.sh/uv/install.sh | sh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv pip install -r requirements.in
uv pip install -r requirements.dev
```

Пример запуска:

```
python .\src\pulsetrace\main.py --cfg config/config.yaml
```

TODOs:
+ 1 keras model
+ 3 tf models
+ arima
SHAP local/global
pytorch model with custom class
png generate / html generate
