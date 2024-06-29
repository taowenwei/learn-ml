import json
import sys


def genType(dtype):
    match dtype:
        case 'integer':
            return 'int'
        case 'string':
            return 'str'
        case _:
            raise RuntimeError(f'unexpected data type {dtype}')


def genToolParameters(methodBody, components):
    def codegen(name, dtype, required=True):
        if required:
            return f'{name}: {genType(dtype)}'
        return f'{name}: Optional[{genType(dtype)}]'

    if 'parameters' in methodBody:
        paramList = list(map(lambda parameter: codegen(
            parameter['name'], parameter['schema']['type'], parameter['required']), methodBody['parameters']))
        return f"{', '.join(paramList)}"

    if 'requestBody' in methodBody:
        schema = methodBody['requestBody']['content']['application/json']['schema']['$ref']
        schema = schema[schema.rfind('/') + 1:]
        schemaDef = components['schemas'][schema]
        properties = list(filter(
            lambda property: property in schemaDef['required'], schemaDef['properties'].keys()))
        paramList = list(map(lambda property: codegen(
            property, schemaDef['properties'][property]['type']), properties))
        return f"{', '.join(paramList)}"

    return ''


def genTool(path, method, methodBody, components):
    print(f'    generating tool for {path} {method}')

    tool = f"""
        @tool
        def {methodBody['operationId']}({genToolParameters(methodBody, components)}) -> :
            '''{methodBody['summary']}'''

            response = requests.{method}(__baseUrl + /{path})
            return response.json()
    """
    lines = tool.split('\n')[1:]
    indent = lines[0].index('@')
    lines = list(map(lambda line: line[indent:], lines))
    return '\n'.join(lines)


def genPath(pathName, pathBody, components):
    print(f'generating {pathName}')

    methods = pathBody.keys()
    tools = list(map(lambda method: genTool(
        pathName, method, pathBody[method], components), methods))
    for tool in tools:
        print(tool)


def convert(apiSpec):
    with open(apiSpec, 'r') as file:
        spec = json.load(file)

        paths = spec['paths']
        components = spec['components']
        toolCode = list(map(lambda key: genPath(
            key, paths[key], components), paths.keys()))


if __name__ == '__main__':
    # tool description and usage
    print('\nopenapi2tool: Transform an OpenAPI specification into LangChain tools\n')
    arguments = sys.argv
    if len(arguments) != 2:
        print('Usage: openapi2tool <your_openapi_spec.json>')
        exit
    convert(arguments[1])
