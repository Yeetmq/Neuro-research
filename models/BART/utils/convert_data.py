from datasets import load_dataset, concatenate_datasets
import aiohttp
from pathlib import Path
from datasets import Dataset
import numpy as np
import gzip
import json
import shutil
from collections import defaultdict

# Глобальные счетчики для сбора статистики
stats = defaultdict(int)

def convert_to_jsonl(input_path, output_path):
    """Конвертирует файл в JSONL формат с обработкой ошибок"""
    local_stats = {'total_lines': 0, 'parsed_lines': 0, 'error_lines': 0}
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            local_stats['total_lines'] += 1
            if not line:
                continue

            try:
                data = json.loads(line)
                json.dump(data, f_out, ensure_ascii=False)
                f_out.write('\n')
                local_stats['parsed_lines'] += 1
            except json.JSONDecodeError:
                if line.startswith('['):
                    try:
                        for item in json.loads(line):
                            json.dump(item, f_out, ensure_ascii=False)
                            f_out.write('\n')
                        local_stats['parsed_lines'] += 1
                    except:
                        local_stats['error_lines'] += 1
                        print(f"Failed to parse array in: {input_path}")
                else:
                    local_stats['error_lines'] += 1
                    print(f"Invalid JSON line skipped in: {input_path}")
            except Exception as e:
                local_stats['error_lines'] += 1
                print(f"Unexpected error: {str(e)}")

    # Обновляем глобальную статистику
    stats['total_lines'] += local_stats['total_lines']
    stats['parsed_lines'] += local_stats['parsed_lines']
    stats['error_lines'] += local_stats['error_lines']
    
    return local_stats

def process_gz_files(source_root, target_root):
    source_path = Path(source_root)
    target_path = Path(target_root)
    file_stats = defaultdict(int)

    for gz_file in source_path.rglob("*.gz"):
        file_stats['total_gz_files'] += 1
        try:
            relative_path = gz_file.relative_to(source_path)
            output_dir = target_path / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            temp_file = output_dir / gz_file.name
            final_file = output_dir / gz_file.name.replace(".gz", ".json")

            with gzip.open(gz_file, 'rb') as f_in:
                with open(temp_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            convert_stats = convert_to_jsonl(temp_file, final_file)
            temp_file.unlink()
            file_stats['success_gz'] += 1
            
            # Логируем статистику по файлу
            print(f"Processed: {gz_file.name}")
            print(f"  Lines: {convert_stats['total_lines']} "
                  f"(OK: {convert_stats['parsed_lines']}, "
                  f"Errors: {convert_stats['error_lines']})")

        except Exception as e:
            file_stats['errors_gz'] += 1
            print(f"Error processing {gz_file}: {str(e)}")
            if temp_file.exists():
                temp_file.unlink()

    # Обновляем глобальную статистику
    stats.update(file_stats)
    print('\nGZ Processing Summary:')
    print(f"Total GZ files: {file_stats['total_gz_files']}")
    print(f"Successfully processed: {file_stats['success_gz']}")
    print(f"Failed: {file_stats['errors_gz']}")
    print('---------------------------')

def filter_and_save_records(source_root, target_root):
    """Фильтрация записей с детальной статистикой"""
    source_path = Path(source_root)
    target_path = Path(target_root)
    filter_stats = defaultdict(int)

    processed_files = set(target_path.rglob("*.json"))
    
    for src_file in source_path.rglob("*.json"):
        filter_stats['total_filtered_files'] += 1
        relative_path = src_file.relative_to(source_path)
        dst_file = target_path / relative_path
        
        if dst_file.exists():
            filter_stats['skipped_files'] += 1
            continue
            
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(src_file, 'r', encoding='utf-8') as f_in, \
                 open(dst_file, 'w', encoding='utf-8') as f_out:

                file_stats = {
                    'total_records': 0,
                    'filtered_records': 0,
                    'error_records': 0
                }
                
                for line in f_in:
                    line = line.strip()
                    file_stats['total_records'] += 1
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        abstract = record.get('abstract', '')
                        
                        if len(abstract.split()) >= 200:
                            json.dump(record, f_out, ensure_ascii=False)
                            f_out.write('\n')
                            file_stats['filtered_records'] += 1
                            
                    except json.JSONDecodeError:
                        file_stats['error_records'] += 1
                    except Exception as e:
                        file_stats['error_records'] += 1
                        print(f"Error processing record: {e}")

                # Обновляем статистику
                filter_stats['total_records'] += file_stats['total_records']
                filter_stats['filtered_records'] += file_stats['filtered_records']
                filter_stats['error_records'] += file_stats['error_records']
                filter_stats['processed_files'] += 1
                
                # Логируем статистику по файлу
                print(f"Filtered: {src_file.name}")
                print(f"  Records: {file_stats['total_records']} "
                      f"(Filtered: {file_stats['filtered_records']}, "
                      f"Errors: {file_stats['error_records']})")

        except Exception as e:
            filter_stats['error_files'] += 1
            print(f"Error processing file {src_file}: {e}")
            if dst_file.exists():
                dst_file.unlink()

    # Обновляем глобальную статистику
    stats.update(filter_stats)
    print('\nFiltering Summary:')
    print(f"Total files: {filter_stats['total_filtered_files']}")
    print(f"Processed: {filter_stats['processed_files']}")
    print(f"Skipped: {filter_stats['skipped_files']}")
    print(f"Error files: {filter_stats['error_files']}")
    print(f"Total records: {filter_stats['total_records']}")
    print(f"Filtered records: {filter_stats['filtered_records']}")
    print(f"Error records: {filter_stats['error_records']}")
    print('---------------------------')

def print_final_stats():
    """Выводит итоговую статистику обработки"""
    print('\n' + '='*50)
    print('FINAL PROCESSING STATISTICS:')
    print('GZ Files Processing:')
    print(f"  Total GZ files: {stats['total_gz_files']}")
    print(f"  Success: {stats['success_gz']}")
    print(f"  Errors: {stats['errors_gz']}")
    
    print('\nJSONL Conversion:')
    print(f"  Total lines: {stats['total_lines']}")
    print(f"  Parsed lines: {stats['parsed_lines']}")
    print(f"  Error lines: {stats['error_lines']}")
    
    print('\nFiltering Process:')
    print(f"  Total files: {stats['total_filtered_files']}")
    print(f"  Processed: {stats['processed_files']}")
    print(f"  Skipped: {stats['skipped_files']}")
    print(f"  Error files: {stats['error_files']}")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Filtered records: {stats['filtered_records']}")
    print(f"  Error records: {stats['error_records']}")
    print('='*50 + '\n')

if __name__ == '__main__':
    # Пример вызова функций
    process_gz_files('source_data', 'converted_data')
    filter_and_save_records('converted_data', 'filtered_data')
    print_final_stats()