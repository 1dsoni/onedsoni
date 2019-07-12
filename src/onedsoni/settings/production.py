

from .base import *
import os
import dj_database_url
db_from_env = dj_database_url.config()
# DATABASES['default'].update(db_from_env)

DEBUG = ( os.getenv('DEBUG', False) == 'True')
ALLOWED_HOSTS = ['herokuapp.com','onedsoni.herokuapp.com',]

SECRET_KEY = os.getenv('SECRET_KEY', '03l%&8d(%j)pu_pvy$@s!m1rm&6rq_^=ui!*c&7=e(9r4anf41')

DATABASES = {
    'default': dj_database_url.config(
        default=os.getenv('DATABASE_URL', 'asdasdashkdghghg7689769uig'),
    )
}
# DATABASES['default']['CONN_MAX_AGE'] = 500

# Static assets
STATIC_ROOT = os.path.join(BASE_DIR, 'static')
STATICFILES_DIRS = [
   os.path.join(BASE_DIR, 'image_toolkit/static'),
   ]

STATICFILES_STORAGE = 'whitenoise.django.GzipManifestStaticFilesStorage'
# User uploads
# MEDIA_ROOT = os.path.join(BASE_DIR, 'static', 'media')

if not DEBUG:
    CORS_REPLACE_HTTPS_REFERER      = True
    HOST_SCHEME                     = "https://"
    SECURE_PROXY_SSL_HEADER         = ('HTTP_X_FORWARDED_PROTO', 'https')
    SECURE_SSL_REDIRECT             = True
    SESSION_COOKIE_SECURE           = True
    CSRF_COOKIE_SECURE              = True
    SECURE_HSTS_INCLUDE_SUBDOMAINS  = True
    SECURE_HSTS_SECONDS             = 1000000
    SECURE_FRAME_DENY               = True
else :
    CORS_REPLACE_HTTPS_REFERER      = False
    HOST_SCHEME                     = "http://"
    SECURE_PROXY_SSL_HEADER         = None
    SECURE_SSL_REDIRECT             = False
    SESSION_COOKIE_SECURE           = False
    CSRF_COOKIE_SECURE              = False
    SECURE_HSTS_SECONDS             = None
    SECURE_HSTS_INCLUDE_SUBDOMAINS  = False
    SECURE_FRAME_DENY               = False

# # add this


# import os
# import json
# from django.core.exceptions import ImproperlyConfigured
#
# with open(os.environ.get('ONEDSONI_CONFIG')) as f:
#  configs = json.loads(f.read())
# def get_env_var(setting, configs=configs):
#  try:
#      val = configs[setting]
#      if val == 'True':
#          val = True
#      elif val == 'False':
#          val = False
#      return val
#  except KeyError:
#      error_msg = "ImproperlyConfigured: Set {0} environment variable".format(setting)
#      raise ImproperlyConfigured(error_msg)
# #get secret key
# SECRET_KEY = get_env_var("SECRET_KEY")
# DB_INFO = get_env_var("DATABASE_URL").split(':')
#
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.postgresql_psycopg2',
#         'NAME': DB_INFO[3].split('/')[1],
#         'USER': DB_INFO[1].split('//')[1],
#         'PASSWORD': DB_INFO[2].split('@')[0],
#         'HOST': DB_INFO[2].split('@')[1],
#         'PORT': DB_INFO[3].split('/')[0],
#     }
# }
