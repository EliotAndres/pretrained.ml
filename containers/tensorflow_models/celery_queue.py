from __future__ import absolute_import, unicode_literals
from celery import Celery
import redis

import config

CELERY_BROKER_URL = 'redis://redis/0'

app = Celery('proj', broker=config.REDIS_URL,
             backend=config.REDIS_URL,
             include=['tasks'])

app.conf.task_serializer = 'pickle'
app.conf.accept_content = ['json', 'pickle']

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
)

redis_instance = redis.StrictRedis(host='redis', port=6379, db=0)

if __name__ == '__main__':
    app.start()

