from __future__ import absolute_import, unicode_literals
from celery import Celery
import redis

CELERY_BROKER_URL = 'redis://localhost'
CELERY_RESULT_BACKEND = 'redis://localhost'

app = Celery('proj', broker=CELERY_BROKER_URL,
             backend=CELERY_RESULT_BACKEND,
             include=['tasks'])

app.conf.task_serializer = 'pickle'
app.conf.accept_content = ['json', 'pickle']

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
)

redis_instance = redis.StrictRedis(host='localhost', port=6379, db=0)

if __name__ == '__main__':
    app.start()

