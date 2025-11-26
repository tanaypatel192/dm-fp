"""
Gunicorn Configuration for Production

Usage:
    gunicorn -c gunicorn.conf.py app:app
"""

import multiprocessing
import os

# Server Socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker Processes
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = os.getenv('WORKER_CLASS', 'uvicorn.workers.UvicornWorker')
worker_connections = int(os.getenv('WORKER_CONNECTIONS', 1000))
max_requests = 1000  # Restart workers after this many requests
max_requests_jitter = 100  # Random jitter to prevent all workers restarting at once
timeout = int(os.getenv('TIMEOUT', 120))
keepalive = int(os.getenv('KEEPALIVE', 5))

# Threading
threads = 2

# Server Mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Logging
accesslog = 'logs/access.log'
errorlog = 'logs/error.log'
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process Naming
proc_name = os.getenv('APP_NAME', 'diabetes-api')

# Server Hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    print(f"Starting Gunicorn server...")
    print(f"Workers: {workers}")
    print(f"Worker class: {worker_class}")
    print(f"Bind: {bind}")


def when_ready(server):
    """Called just after the server is started."""
    print("Server is ready. Spawning workers")


def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    print("Reloading...")


def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    print(f"Worker {worker.pid} was interrupted")


def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    print(f"Worker {worker.pid} received SIGABRT")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    print(f"Worker spawned (pid: {worker.pid})")


def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    print(f"Worker initialized (pid: {worker.pid})")


def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    print(f"Worker exited (pid: {worker.pid})")


def nworkers_changed(server, new_value, old_value):
    """Called just after number of workers has been changed."""
    print(f"Number of workers changed from {old_value} to {new_value}")


def on_exit(server):
    """Called just before exiting Gunicorn."""
    print("Shutting down Gunicorn")


# SSL/HTTPS (if using Gunicorn directly without nginx)
# Uncomment and configure if needed
# keyfile = '/path/to/keyfile.key'
# certfile = '/path/to/certfile.crt'
# ssl_version = 'TLSv1_2'
# cert_reqs = 0
# do_handshake_on_connect = False
# ciphers = 'TLS_AES_256_GCM_SHA384'
