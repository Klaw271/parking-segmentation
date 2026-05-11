import os

# Простая транслитерация кириллицы
CYRILLIC_TO_LATIN = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo', 'ж': 'zh',
    'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o',
    'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'ts',
    'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'
}

def transliterate(text):
    result = []
    for char in text.lower():
        if char in CYRILLIC_TO_LATIN:
            result.append(CYRILLIC_TO_LATIN[char])
        else:
            result.append(char)
    return ''.join(result)

MASK_DIR = 'data/self_made_dataset/masks'

# Переименование масок
print("Renaming masks with Cyrillic characters...")
count = 0
for filename in os.listdir(MASK_DIR):
    old_path = os.path.join(MASK_DIR, filename)
    if not os.path.isfile(old_path):
        continue
    
    # Проверяем, есть ли кириллица
    if any(ord(c) > 127 for c in filename):
        name, ext = os.path.splitext(filename)
        new_name = transliterate(name) + ext
        new_path = os.path.join(MASK_DIR, new_name)
        
        os.rename(old_path, new_path)
        print(f"  {filename} -> {new_name}")
        count += 1

print(f"Renamed {count} mask files")
