#!/bin/sh

(cd /opt/srv/tensorflow_models/tensorflow_models_repo/research \
 && protoc object_detection/protos/*.proto --python_out=. )

su -m celery-user -c "celery -A celery_queue worker -E -l info --concurrency=1"
