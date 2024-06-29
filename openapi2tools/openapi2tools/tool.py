import json
import sys

def convert(apiSpec):
    spec = json.load(apiSpec)


if __name__ == '__main__':
    # tool description and usage
    print('\nopenapi2tool: Transform an OpenAPI specification into LangChain tools\n')
    arguments = sys.argv
    if len(arguments) != 2:
        print('Usage: openapi2tool <your_openapi_spec.json>')
        exit
    convert(arguments[1])
