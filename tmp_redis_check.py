import redis

try:
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    print('Redis broker ping ->', r.ping())
    print("Length celery queue ->", r.llen('celery'))
except Exception as e:
    print('Redis check error:', e)
