METADATA =\
{
	'name': 'faceweave',
	'description': 'Next generation face swapper and enhancer',
	'version': '2.1.1',
	'license': 'MIT',
	'author': 'Henry Ruhs',
	'url': 'https://faceweave.io'
}


def get(key : str) -> str:
	return METADATA[key]
