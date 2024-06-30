import json
import sys


def indentCode(indent, lines):
    # only indent from the 2nd line
    indented = [lines[0],  *
            (list(map(lambda line: ' ' * indent + line, lines[1:])))]
    return '\n'.join(indented)


def genType(dtype):
    match dtype:
        case 'integer':
            return 'int'
        case 'string':
            return 'str'
        case 'array':
            return 'list'
        case 'boolean':
            return 'bool'
        case 'number':
            return 'float'
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


def genToolReturn(methodBody):
    schema = methodBody['responses']['200']['content']['application/json']['schema']
    if 'type' in schema:
        return genType(schema['type'])
    if '$ref' in schema:
        return 'dict'
    raise RuntimeError(f'unexpected tool return data type {schema}')


def genHttpPath(path, methodBody):
    if 'parameters' in methodBody:
        parameters = list(
            filter(lambda parameter: parameter['in'] == 'query', methodBody['parameters']))
        queries = list(map(lambda parameter:
                           "f'" + parameter['name'] + '={' + parameter['name'] + '}' + '\' if ' + parameter['name'] + " != None else ''", parameters))
        queries = list(filter(lambda query: query != '', queries))
        queries = list(map(lambda query: f'({query})', queries))
        if len(queries) > 0:
            return f"'{path}?' + " + " + '&' + ".join(queries)
    return f"'{path}'"


def genHttpBody(indent, methodBody, components):
    if 'requestBody' in methodBody and methodBody['requestBody']['required']:
        schema = methodBody['requestBody']['content']['application/json']['schema']['$ref']
        schema = schema[schema.rfind('/') + 1:]
        schemaDef = components['schemas'][schema]
        properties = list(filter(
            lambda property: property in schemaDef['required'], schemaDef['properties'].keys()))
        paramList = list(
            map(lambda property: f"{' ' * 4}'{property}': {property}, ", properties))
        lines = ['', 'data = {', *paramList, '}']
        return indentCode(indent, lines)
    return ''


def genSuccess(indent):
    return ''


def genFailure(indent):
    return ''


def genTool(clzName, path, method, methodBody, components):
    print(f'    generating tool for {path} {method}')

    tool = f"""
    @tool
    def {methodBody['operationId']}({genToolParameters(methodBody, components)}) -> {genToolReturn(methodBody)}:
        '''{methodBody['summary']}'''
        {genHttpBody(8, methodBody, components)}
        response = requests.{method}({clzName}.BaseUrl + {genHttpPath(path, methodBody)}, headers={clzName}.HttpHeader{', json=data' if 'requestBody' in methodBody and methodBody['requestBody']['required'] else ''})
        {genSuccess(8)}
        {genFailure(8)}
        return response.json()
    """
    lines = tool.split('\n')[1:]
    return '\n'.join(lines)


def genPath(clzName, pathName, pathBody, components):
    print(f'generating {pathName}')

    methods = pathBody.keys()
    tools = list(map(lambda method: genTool(
        clzName, pathName, method, pathBody[method], components), methods))
    return tools


def genToolCapabilities(indent, paths):
    def genCapability(pathBody):
        methods = pathBody.keys()
        return list(map(lambda method: f"+ use the `{pathBody[method]['operationId']}` tool to {pathBody[method]['summary'].lower()}", methods))

    capabilities = list(map(lambda key: genCapability(
        paths[key]), paths.keys()))
    capabilities = sum(capabilities, [])
    return indentCode(indent, capabilities)


def convert(apiSpec):
    spec = None
    with open(apiSpec, 'r') as file:
        spec = json.load(file)

    paths = spec['paths']
    components = spec['components']
    clzName = spec['info']['title'] + 'Api'

    tools = list(map(lambda key: genPath(
        clzName, key, paths[key], components), paths.keys()))
    tools = sum(tools, [])
    toolCode = '\n'.join(tools)

    clzBody = f"""
from typing import Optional
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

class {clzName}:

    BaseUrl = 'http://localhost:4000'

    HttpHeader = {{
            'Content-Type': 'application/json',
            'Authorization': 'Bearer YOUR_ACCESS_TOKEN'
        }}

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            '''You are a specialized {spec['info']['description']}. You can,

            {genToolCapabilities(12, paths)}
            
            Now answer your question'''
        ),
        ('user', '{{user}}'),
    ])
    """

    lines = clzBody.split('\n')[1:]
    with open('output/abcdef.py', 'w') as wf:
        wf.write('\n'.join(lines) + '\n' + toolCode)


if __name__ == '__main__':
    # tool description and usage
    print('\nopenapi2tool: Transform an OpenAPI specification into LangChain tools\n')
    arguments = sys.argv
    if len(arguments) != 2:
        print('Usage: openapi2tool <your_openapi_spec.json>')
        exit
    convert(arguments[1])
