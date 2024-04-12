# here the tests for the api endpoints
# check status_code, correct message, server live check first

import requests

# definition of the API address, defined by service name of the fastapi container in the docker-compose file
api_address = 'localhost'


# API port, defined in the docker compose file
api_port = 8000


r_live = requests.get(
    url='http://{address}:{port}/'.format(address=api_address, port=api_port)
)

# in jedem test als erstes: assert r_live.status_code == 200

print(r_live.status_code)