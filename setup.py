from setuptools import setup, find_packages

setup(
    name='directRetrieval',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'httpx',
        'requests',
    ],
    author='Cristian Desivo',
    author_email='cdesivo92@gmail.com',
    description='A package to use LLMs to retrieve information',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/crisdesivo/directRetrieval',
    classifiers=[
        'Programming Language :: Python :: 3',
        # 'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)