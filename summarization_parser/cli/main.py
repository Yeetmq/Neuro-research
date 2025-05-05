import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from core.parse import SummarizationParser
from core.utils import load_config

def main():
    parser = argparse.ArgumentParser(description='Data parser for BART summarization')
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--config', default='config/settings.yaml', help='Config file path')
    parser.add_argument('--translate', default='false', help='Need translation?')

    sample_agrs_list = [
        # '--query', r'Transformers in machine learning',
        '--query', r'Трансформеры в машинном обучении',
        '--config', r'D:\ethd\ml\Neuro-research\summarization_parser\config\settings.yaml',
        '--translate', 'false'
    ]

    args = parser.parse_args(sample_agrs_list)
    
    config = load_config(args.config)
    config['query'] = args.query
    config['translate'] = args.translate
    print(config['query'])
    parser = SummarizationParser(config)
    parser.run()

if __name__ == '__main__':
    main()