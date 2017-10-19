#!/bin/sh

su -m celery-user -c "celery -A celery_queue worker -E -l info --concurrency=1"
