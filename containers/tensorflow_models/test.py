import socketio

sio = socketio.Server()

@sio.on('connect')
def connect(sid, environ):
    print('connect ', sid)
