# https://pythonspeed.com/articles/base-image-python-docker-images/
FROM python:3.11-slim-bookworm

# Виставляємо робочу директорію контейнера в /app
WORKDIR /ml_server

# Копіюємо файл з залежностями, створюється новий шар
COPY requirements.txt requirements.txt

# Встановлюємо залежності вказалі requirements.txt - теж новий шор
RUN pip install --no-cache-dir -r requirements.txt

# Додаємо наш код /app - створили новий шар. таким чином при ребілді не потрібно заново ставити залежності при зміні коду
ADD . /ml_server

# Документуємо порт, що використовується контейнером. В компоузі буде юзатися для міжконтейнерного зв'язку
EXPOSE 6060:5000

# Run ml_server.py when the container launches
# https://spacelift.io/blog/docker-entrypoint-vs-cmd
ENTRYPOINT [ "python" ] 
CMD ["-u", "ml_server.py"]
